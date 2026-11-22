//! Build one ABMESH01 frame from a local v3.3 database (debug/acceptance helper).

use std::process::ExitCode;

use alphabuilder_studio_lib::commands::datasets::{build_local_frame, frame_data_for_ref};
use alphabuilder_studio_lib::commands::frame::build_abmesh;
use alphabuilder_studio_lib::contracts::{DataSourceRef, DbKind, MeshOptions};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: render_frame <db> <entry> <step> <output.bin>");
        return ExitCode::from(2);
    }
    let source = DataSourceRef {
        origin: "local".to_string(),
        db_id: args[1].clone(),
        kind: DbKind::HarvestV33,
    };
    let step: u32 = match args[3].parse() {
        Ok(value) => value,
        Err(error) => {
            eprintln!("invalid step: {error}");
            return ExitCode::from(2);
        }
    };
    let options = MeshOptions::default();
    let data = match frame_data_for_ref(&source, &args[2], step, &options) {
        Ok(data) => data,
        Err(error) => {
            eprintln!("frame error: {error}");
            return ExitCode::from(3);
        }
    };
    let bytes = match build_abmesh(&data, &options) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("abmesh error: {error}");
            return ExitCode::from(4);
        }
    };
    if let Err(error) = std::fs::write(&args[4], &bytes) {
        eprintln!("write error: {error}");
        return ExitCode::from(5);
    }
    println!(
        "step={} bytes={} policy={}+{} diff={}+{}",
        step,
        bytes.len(),
        data.policy_add.iter().filter(|&&v| v > 0.0).count(),
        data.policy_remove.iter().filter(|&&v| v > 0.0).count(),
        data.diff_added.len(),
        data.diff_removed.len()
    );
    let _ = build_local_frame;
    ExitCode::SUCCESS
}
