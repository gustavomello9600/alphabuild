//! Cross-language verifier used by the Python test-suite.
//!
//! Usage: cargo run -p ab-core --example abwire_verify -- <packet.bin>

use std::process::ExitCode;

use ab_core::codecs::wire::{AbMeshPacket, WirePacket};

fn main() -> ExitCode {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: abwire_verify <packet.bin>");
        return ExitCode::from(2);
    };
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(err) => {
            eprintln!("read error: {err}");
            return ExitCode::from(2);
        }
    };
    if bytes.len() >= 8 && &bytes[..8] == b"ABMESH01" {
        return match AbMeshPacket::decode(&bytes) {
            Ok(packet) => {
                println!(
                    "{{\"protocol\":\"ABMESH01\",\"step\":{},\"vertices\":{},\"indices\":{}}}",
                    packet.step_index,
                    packet.voxel_ids.len(),
                    packet.indices.len()
                );
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("ABMESH01 decode error: {err}");
                ExitCode::from(3)
            }
        };
    }
    match WirePacket::decode(&bytes) {
        Ok(packet) => {
            let tags: Vec<u16> = packet.tensors.iter().map(|t| t.tag).collect();
            println!(
                "{{\"protocol\":\"ABWIRE01\",\"payload_type\":{},\"tensors\":{:?},\"header\":{}}}",
                packet.payload_type, tags, packet.header_json
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("ABWIRE01 decode error: {err}");
            ExitCode::from(3)
        }
    }
}
