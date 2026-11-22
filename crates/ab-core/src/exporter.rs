//! Native exporters: Binary STL, Wavefront OBJ and VTK ImageData (`.vti`).

use std::fmt::Write as _;

use crate::coords::Dims;
use crate::error::{CoreError, CoreResult};
use crate::mesher::Mesh;

/// Exact 80-byte STL header.
pub const STL_HEADER: &[u8] = b"AlphaBuilder Studio v4.0 Binary STL";

/// Export a watertight binary STL from the surface-culled mesh.
pub fn export_stl(mesh: &Mesh) -> CoreResult<Vec<u8>> {
    if !mesh.positions.len().is_multiple_of(3)
        || mesh.normals.len() != mesh.positions.len()
        || !mesh.indices.len().is_multiple_of(3)
    {
        return Err(CoreError::InvalidInput {
            message: "malformed mesh for STL export".to_string(),
        });
    }
    let triangle_count = mesh.indices.len() / 3;
    let mut out = Vec::with_capacity(84 + triangle_count * 50);
    let mut header = [0_u8; 80];
    let copy_len = STL_HEADER.len().min(80);
    header[..copy_len].copy_from_slice(&STL_HEADER[..copy_len]);
    out.extend_from_slice(&header);
    out.extend_from_slice(&(triangle_count as u32).to_le_bytes());
    for tri in mesh.indices.as_chunks::<3>().0 {
        let normal_offset = tri[0] as usize * 3;
        let normal = [
            mesh.normals[normal_offset],
            mesh.normals[normal_offset + 1],
            mesh.normals[normal_offset + 2],
        ];
        for v in normal {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &idx in tri {
            let offset = idx as usize * 3;
            let x = *mesh
                .positions
                .get(offset)
                .ok_or_else(|| CoreError::InvalidInput {
                    message: "STL index out of range".to_string(),
                })?;
            let y = *mesh
                .positions
                .get(offset + 1)
                .ok_or_else(|| CoreError::InvalidInput {
                    message: "STL index out of range".to_string(),
                })?;
            let z = *mesh
                .positions
                .get(offset + 2)
                .ok_or_else(|| CoreError::InvalidInput {
                    message: "STL index out of range".to_string(),
                })?;
            out.extend_from_slice(&x.to_le_bytes());
            out.extend_from_slice(&y.to_le_bytes());
            out.extend_from_slice(&z.to_le_bytes());
        }
        out.extend_from_slice(&0_u16.to_le_bytes());
    }
    debug_assert_eq!(out.len(), 84 + triangle_count * 50);
    Ok(out)
}

/// Export a Wavefront OBJ with `v`, `vn` and `f i//i` faces.
pub fn export_obj(mesh: &Mesh, object_name: &str) -> CoreResult<String> {
    if !mesh.positions.len().is_multiple_of(3)
        || mesh.normals.len() != mesh.positions.len()
        || !mesh.indices.len().is_multiple_of(3)
    {
        return Err(CoreError::InvalidInput {
            message: "malformed mesh for OBJ export".to_string(),
        });
    }
    let name = object_name.replace(['\n', '\r'], "_");
    let mut out = String::new();
    out.push_str("# AlphaBuilder Studio v4.0\n");
    let _ = writeln!(out, "o {name}");
    for chunk in mesh.positions.as_chunks::<3>().0 {
        let _ = writeln!(out, "v {:.6} {:.6} {:.6}", chunk[0], chunk[1], chunk[2]);
    }
    for chunk in mesh.normals.as_chunks::<3>().0 {
        let _ = writeln!(out, "vn {:.6} {:.6} {:.6}", chunk[0], chunk[1], chunk[2]);
    }
    let vertex_count = mesh.positions.len() / 3;
    for tri in mesh.indices.as_chunks::<3>().0 {
        let a = tri[0] as usize + 1;
        let b = tri[1] as usize + 1;
        let c = tri[2] as usize + 1;
        if a > vertex_count || b > vertex_count || c > vertex_count {
            return Err(CoreError::InvalidInput {
                message: "OBJ index out of range".to_string(),
            });
        }
        let _ = writeln!(out, "f {a}//{a} {b}//{b} {c}//{c}");
    }
    Ok(out)
}

/// Scalar fields exported to ParaView.
pub struct VtiFields<'a> {
    pub density: &'a [f32],
    pub displacement: Option<&'a [f32]>,
    pub strain: Option<&'a [f32]>,
    pub blueprint: Option<&'a [f32]>,
}

fn write_data_array(
    out: &mut String,
    name: &str,
    dims: Dims,
    values: Option<&[f32]>,
) -> CoreResult<()> {
    let _ = writeln!(
        out,
        "        <DataArray type=\"Float32\" Name=\"{name}\" NumberOfComponents=\"1\" format=\"ascii\">"
    );
    let zeros = vec![0.0_f32; dims.len()];
    let data = values.unwrap_or(&zeros);
    if data.len() != dims.len() {
        return Err(CoreError::InvalidInput {
            message: format!("VTI field '{name}' length mismatch"),
        });
    }
    // VTK ImageData expects x-fastest ordering; our arrays are C-order (x slowest).
    for z in 0..dims.w {
        for y in 0..dims.h {
            for x in 0..dims.d {
                let idx = x as usize * dims.plane_hw() + y as usize * dims.w as usize + z as usize;
                let _ = write!(out, "          {:.7e}", data[idx]);
                out.push('\n');
            }
        }
    }
    out.push_str("        </DataArray>\n");
    Ok(())
}

/// Export a valid VTK ImageData XML file with the four scientific fields.
pub fn export_vti(dims: Dims, fields: &VtiFields<'_>) -> CoreResult<String> {
    dims.validate()?;
    if fields.density.len() != dims.len() {
        return Err(CoreError::InvalidInput {
            message: "density field length mismatch".to_string(),
        });
    }
    let extent = format!("0 {} 0 {} 0 {}", dims.d - 1, dims.h - 1, dims.w - 1);
    let mut out = String::new();
    out.push_str("<?xml version=\"1.0\"?>\n");
    out.push_str(
        "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt32\">\n",
    );
    let _ = writeln!(
        out,
        "  <ImageData WholeExtent=\"{extent}\" Origin=\"0 0 0\" Spacing=\"1 1 1\">"
    );
    let _ = writeln!(out, "    <Piece Extent=\"{extent}\">");
    out.push_str("      <PointData Scalars=\"density\">\n");
    write_data_array(&mut out, "density", dims, Some(fields.density))?;
    write_data_array(&mut out, "displacement", dims, fields.displacement)?;
    write_data_array(&mut out, "strain", dims, fields.strain)?;
    write_data_array(&mut out, "blueprint", dims, fields.blueprint)?;
    out.push_str("      </PointData>\n");
    out.push_str("      <CellData>\n      </CellData>\n");
    out.push_str("    </Piece>\n");
    out.push_str("  </ImageData>\n");
    out.push_str("</VTKFile>\n");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesher::build_mesh;

    fn cube_2x2x2() -> Mesh {
        let dims = Dims::new(2, 2, 2);
        build_mesh(&vec![1.0; dims.len()], dims, 0.5, None, None).unwrap()
    }

    #[test]
    fn binary_stl_size_is_exact() {
        let stl = export_stl(&cube_2x2x2()).unwrap();
        assert_eq!(stl.len(), 84 + 48 * 50);
        assert_eq!(stl.len(), 2484);
        assert_eq!(&stl[..STL_HEADER.len()], STL_HEADER);
        assert!(stl[STL_HEADER.len()..80].iter().all(|&b| b == 0));
        let count = u32::from_le_bytes([stl[80], stl[81], stl[82], stl[83]]);
        assert_eq!(count, 48);
    }

    #[test]
    fn obj_contains_one_face_per_triangle() {
        let obj = export_obj(&cube_2x2x2(), "cube").unwrap();
        assert_eq!(obj.lines().filter(|l| l.starts_with("v ")).count(), 96);
        assert_eq!(obj.lines().filter(|l| l.starts_with("vn ")).count(), 96);
        assert_eq!(obj.lines().filter(|l| l.starts_with("f ")).count(), 48);
        assert!(obj.lines().any(|l| l == "o cube"));
    }

    #[test]
    fn vti_is_valid_xml_with_all_fields() {
        let dims = Dims::new(4, 2, 2);
        let density = vec![1.0_f32; dims.len()];
        let vti = export_vti(
            dims,
            &VtiFields {
                density: &density,
                displacement: None,
                strain: None,
                blueprint: None,
            },
        )
        .unwrap();
        assert!(vti.starts_with("<?xml version=\"1.0\"?>"));
        assert!(vti.contains("<VTKFile type=\"ImageData\""));
        assert!(vti.contains("WholeExtent=\"0 3 0 1 0 1\""));
        for field in ["density", "displacement", "strain", "blueprint"] {
            assert!(vti.contains(&format!("Name=\"{field}\"")));
        }
        let values = vti
            .lines()
            .filter(|l| l.trim().starts_with("1.0000000e0"))
            .count();
        assert_eq!(values, dims.len());
    }

    #[test]
    fn empty_mesh_stl_is_84_bytes() {
        let dims = Dims::new(3, 3, 3);
        let mesh = build_mesh(&vec![0.0; dims.len()], dims, 0.5, None, None).unwrap();
        let stl = export_stl(&mesh).unwrap();
        assert_eq!(stl.len(), 84);
        assert_eq!(u32::from_le_bytes([stl[80], stl[81], stl[82], stl[83]]), 0);
    }

    #[test]
    fn reject_mismatched_vti_field() {
        let dims = Dims::new(4, 2, 2);
        let density = vec![1.0_f32; 3];
        let err = export_vti(
            dims,
            &VtiFields {
                density: &density,
                displacement: None,
                strain: None,
                blueprint: None,
            },
        );
        assert!(err.is_err());
    }
}
