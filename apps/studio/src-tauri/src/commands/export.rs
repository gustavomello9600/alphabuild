//! Native STL / OBJ / VTI export and 4K PNG saving.

use std::path::Path;

use ab_core::coords::Dims;
use ab_core::exporter::{export_obj, export_stl, export_vti, VtiFields};
use ab_core::mesher::build_mesh;
use ab_core::topology::compute_strain_and_p99;
use tauri::State;

use crate::commands::datasets::frame_data_for_ref;
use crate::commands::frame::{build_abmesh, FrameData};
use crate::contracts::{ExportResult, ExportSource, MeshOptions, StudioVoxelState};
use crate::state::AppState;

fn decode_base64(input: &str) -> Result<Vec<u8>, String> {
    let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut lookup = [255_u8; 256];
    for (idx, &byte) in alphabet.iter().enumerate() {
        lookup[byte as usize] = idx as u8;
    }
    let clean: Vec<u8> = input
        .bytes()
        .filter(|byte| !byte.is_ascii_whitespace())
        .collect();
    let data = clean
        .strip_suffix(b"==")
        .map(|rest| (rest, 2_usize))
        .or_else(|| clean.strip_suffix(b"=").map(|rest| (rest, 1_usize)))
        .unwrap_or((clean.as_slice(), 0));
    let mut out = Vec::with_capacity(data.0.len() * 3 / 4);
    let mut buffer = 0_u32;
    let mut bits = 0_u32;
    for &byte in data.0 {
        let value = lookup[byte as usize];
        if value == 255 {
            return Err("invalid base64 PNG payload".to_string());
        }
        buffer = (buffer << 6) | value as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buffer >> bits) as u8);
        }
    }
    if data.1 == 1 && out.pop().is_none() {
        return Err("truncated base64 payload".to_string());
    }
    Ok(out)
}

fn data_from_studio(state: &StudioVoxelState) -> Result<FrameData, String> {
    FrameData::empty_studio(state).map_err(|e| e.to_string())
}

fn export_frame(data: &FrameData, format: &str) -> Result<Vec<u8>, String> {
    let dims = data.dims;
    let options = MeshOptions::default();
    match format {
        "stl" | "obj" => {
            let strain = match &data.displacement {
                Some(disp) if disp.len() == dims.len() => {
                    compute_strain_and_p99(&data.density, disp, dims).map_err(|e| e.to_string())?
                }
                _ => compute_strain_and_p99(&data.density, &vec![0.0; dims.len()], dims)
                    .map_err(|e| e.to_string())?,
            };
            let mesh = build_mesh(
                &data.density,
                dims,
                options.density_threshold,
                Some(&strain.disp_norm),
                Some(&strain.strain_norm),
            )
            .map_err(|e| e.to_string())?;
            if format == "stl" {
                export_stl(&mesh).map_err(|e| e.to_string())
            } else {
                export_obj(&mesh, "alphabuilder_studio")
                    .map(|text| text.into_bytes())
                    .map_err(|e| e.to_string())
            }
        }
        "vti" => {
            let original_shape = Dims::new(dims.d, dims.h, dims.w);
            let strain = match &data.displacement {
                Some(disp) if disp.len() == original_shape.len() => {
                    compute_strain_and_p99(&data.density, disp, original_shape)
                        .map_err(|e| e.to_string())?
                }
                _ => compute_strain_and_p99(
                    &data.density,
                    &vec![0.0; original_shape.len()],
                    original_shape,
                )
                .map_err(|e| e.to_string())?,
            };
            export_vti(
                original_shape,
                &VtiFields {
                    density: &data.density,
                    displacement: data.displacement.as_deref(),
                    strain: Some(&strain.strain),
                    blueprint: data.blueprint.as_deref(),
                },
            )
            .map(|text| text.into_bytes())
            .map_err(|e| e.to_string())
        }
        other => Err(format!("unsupported export format: {other}")),
    }
}

#[tauri::command(rename_all = "snake_case")]
pub async fn export_geometry(
    _state: State<'_, AppState>,
    source: ExportSource,
    format: String,
    output_path: String,
) -> Result<ExportResult, String> {
    let data = match source {
        ExportSource::Frame {
            source,
            entry_id,
            step,
        } => {
            if source.origin == "remote" {
                return Err(
                    "Exportação remota direta não suportada: abra o frame e exporte localmente."
                        .to_string(),
                );
            }
            frame_data_for_ref(&source, &entry_id, step, &MeshOptions::default())?
        }
        ExportSource::Studio { state } => data_from_studio(&state)?,
    };
    if format != "vti" {
        // Validate that the mesh can be built before writing anything.
        let _ = build_abmesh(&data, &MeshOptions::default()).map_err(|e| e.to_string())?;
    }
    let bytes = export_frame(&data, &format)?;
    let path = Path::new(&output_path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(path, &bytes).map_err(|e| e.to_string())?;
    Ok(ExportResult {
        ok: true,
        bytes_written: bytes.len(),
        output_path,
    })
}

#[tauri::command(rename_all = "snake_case")]
pub async fn export_png_bytes(
    png_base64: String,
    output_path: String,
) -> Result<ExportResult, String> {
    let cleaned = png_base64
        .split_once(',')
        .map(|(_, payload)| payload)
        .unwrap_or(png_base64.as_str());
    let bytes = decode_base64(cleaned)?;
    if bytes.len() < 8 || &bytes[..8] != b"\x89PNG\r\n\x1a\n" {
        return Err("payload is not a PNG image".to_string());
    }
    let path = Path::new(&output_path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(path, &bytes).map_err(|e| e.to_string())?;
    Ok(ExportResult {
        ok: true,
        bytes_written: bytes.len(),
        output_path,
    })
}

#[cfg(test)]
mod tests {
    use super::decode_base64;

    #[test]
    fn base64_png_round_trip_prefix() {
        let bytes = b"\x89PNG\r\n\x1a\nhello";
        let encoded = "iVBORw0KGgpoZWxsbw==";
        assert_eq!(decode_base64(encoded).unwrap(), bytes);
    }
}
