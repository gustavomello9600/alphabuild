//! Read-only SQLite access with dynamic schema introspection.
//!
//! Every local database is opened with `SQLITE_OPEN_READ_ONLY` and
//! `SQLITE_OPEN_NO_MUTEX`, so the Studio can inspect a live WAL database
//! written by harvest/self-play workers without risking corruption.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use rusqlite::types::ValueRef;
use rusqlite::{Connection, OpenFlags, ToSql};
use serde::{Deserialize, Serialize};

use crate::codecs::coo1::Coo1;
use crate::codecs::npy::{parse_npy, DecodedArray};
use crate::codecs::zlib_decompress;
use crate::error::{CoreError, CoreResult};

/// Maximum decompressed size accepted for a single SQLite blob (256 MiB).
pub const MAX_BLOB_BYTES: usize = 256 * 1024 * 1024;

/// Detected database family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DbKind {
    HarvestV33,
    HarvestV2,
    #[serde(rename = "selfplay")]
    SelfPlay,
    LegacyV1,
}

impl DbKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            DbKind::HarvestV33 => "harvest_v33",
            DbKind::HarvestV2 => "harvest_v2",
            DbKind::SelfPlay => "selfplay",
            DbKind::LegacyV1 => "legacy_v1",
        }
    }
}

/// A decoded SQLite blob.
#[derive(Debug, Clone, PartialEq)]
pub enum DecodedBlob {
    Dense(DecodedArray),
    Sparse(Coo1),
}

/// Descriptor shown by the Library drawer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseInfo {
    pub db_id: String,
    pub name: String,
    pub path_display: String,
    pub kind: DbKind,
    pub item_count: i64,
    pub total_steps: i64,
    pub size_mb: f64,
    pub has_legacy_pickle: bool,
}

/// A read-only view over one SQLite database.
pub struct SqliteReader {
    path: PathBuf,
    conn: Connection,
}

impl SqliteReader {
    /// Open with read-only + no-mutex flags and a 5 s busy timeout.
    pub fn open(path: impl AsRef<Path>) -> CoreResult<Self> {
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            return Err(CoreError::NotFound {
                what: path.display().to_string(),
            });
        }
        let conn = Connection::open_with_flags(
            &path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        conn.busy_timeout(Duration::from_millis(5_000))?;
        conn.execute_batch("PRAGMA busy_timeout = 5000;")?;
        Ok(Self { path, conn })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn path_display(&self) -> String {
        self.path.display().to_string()
    }

    pub fn connection(&self) -> &Connection {
        &self.conn
    }

    /// All table names in the database.
    pub fn tables(&self) -> CoreResult<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn has_table(&self, table: &str) -> CoreResult<bool> {
        let mut stmt = self
            .conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name = ?1 LIMIT 1")?;
        let mut rows = stmt.query([table])?;
        Ok(rows.next()?.is_some())
    }

    /// `PRAGMA table_info(<table>)` with identifier validation.
    pub fn columns(&self, table: &str) -> CoreResult<Vec<String>> {
        validate_identifier(table)?;
        let mut stmt = self.conn.prepare(&format!("PRAGMA table_info({table})"))?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn has_column(&self, table: &str, column: &str) -> CoreResult<bool> {
        Ok(self.columns(table)?.iter().any(|c| c == column))
    }

    /// Build a `SELECT` list replacing missing optional columns with `NULL`.
    pub fn select_with_nulls(&self, table: &str, wanted: &[&str]) -> CoreResult<String> {
        validate_identifier(table)?;
        let existing = self.columns(table)?;
        let parts: Vec<String> = wanted
            .iter()
            .map(|name| {
                validate_identifier(name).map(|_| {
                    if existing.iter().any(|c| c == name) {
                        (*name).to_string()
                    } else {
                        format!("NULL AS {name}")
                    }
                })
            })
            .collect::<CoreResult<Vec<_>>>()?;
        Ok(parts.join(", "))
    }

    /// Detect the schema family using only `sqlite_master` and one count.
    pub fn detect_kind(&self) -> CoreResult<DbKind> {
        let tables = self.tables()?;
        let has = |name: &str| tables.iter().any(|t| t == name);

        if has("games") && has("game_steps") {
            return Ok(DbKind::SelfPlay);
        }
        if has("episodes") && has("records") {
            let count: i64 = self
                .conn
                .query_row("SELECT COUNT(*) FROM episodes", [], |row| row.get(0))?;
            if count > 0 {
                let has_v33 = self.has_column("episodes", "v_constructed_blob")?;
                return Ok(if has_v33 {
                    DbKind::HarvestV33
                } else {
                    DbKind::HarvestV2
                });
            }
        }
        if has("training_data") {
            return Ok(DbKind::LegacyV1);
        }
        if has("episodes") && has("records") {
            let has_v33 = self.has_column("episodes", "v_constructed_blob")?;
            return Ok(if has_v33 {
                DbKind::HarvestV33
            } else {
                DbKind::HarvestV2
            });
        }
        Err(CoreError::Unsupported {
            message: format!(
                "unrecognized AlphaBuilder schema (tables: {})",
                tables.join(", ")
            ),
        })
    }

    pub fn item_count(&self, kind: DbKind) -> CoreResult<i64> {
        match kind {
            DbKind::HarvestV33 | DbKind::HarvestV2 => {
                self.conn
                    .query_row("SELECT COUNT(*) FROM episodes", [], |r| r.get(0))
            }
            DbKind::SelfPlay => self
                .conn
                .query_row("SELECT COUNT(*) FROM games", [], |r| r.get(0)),
            DbKind::LegacyV1 => self.conn.query_row(
                "SELECT COUNT(DISTINCT episode_id) FROM training_data",
                [],
                |r| r.get(0),
            ),
        }
        .map_err(CoreError::from)
    }

    pub fn total_steps(&self, kind: DbKind) -> CoreResult<i64> {
        match kind {
            DbKind::HarvestV33 | DbKind::HarvestV2 => {
                self.conn
                    .query_row("SELECT COUNT(*) FROM records", [], |r| r.get(0))
            }
            DbKind::SelfPlay => self
                .conn
                .query_row("SELECT COUNT(*) FROM game_steps", [], |r| r.get(0)),
            DbKind::LegacyV1 => {
                self.conn
                    .query_row("SELECT COUNT(*) FROM training_data", [], |r| r.get(0))
            }
        }
        .map_err(CoreError::from)
    }

    /// Full descriptor for the UI (never mutates the database).
    pub fn describe(&self) -> CoreResult<DatabaseInfo> {
        let kind = self.detect_kind()?;
        let item_count = self.item_count(kind)?;
        let total_steps = self.total_steps(kind)?;
        let size_mb = std::fs::metadata(&self.path)
            .map(|m| m.len() as f64 / (1024.0 * 1024.0))
            .unwrap_or(0.0);
        let name = self
            .path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| self.path_display());
        Ok(DatabaseInfo {
            db_id: self.path_display(),
            name,
            path_display: self.path_display(),
            kind,
            item_count,
            total_steps,
            size_mb,
            has_legacy_pickle: self.detect_legacy_pickle(kind).unwrap_or(false),
        })
    }

    /// Inspect a bounded sample of blobs for the legacy pickle marker.
    pub fn detect_legacy_pickle(&self, kind: DbKind) -> CoreResult<bool> {
        let (table, column) = match kind {
            DbKind::SelfPlay => ("game_steps", "density_blob"),
            DbKind::HarvestV33 | DbKind::HarvestV2 => ("records", "density_blob"),
            DbKind::LegacyV1 => ("training_data", "state_blob"),
        };
        if !self.has_column(table, column)? {
            return Ok(false);
        }
        let sql = format!("SELECT {column} FROM {table} WHERE {column} IS NOT NULL LIMIT 32");
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map([], |row| row.get::<_, Vec<u8>>(0))?;
        for row in rows {
            let blob = row?;
            if let Ok(raw) = zlib_decompress(&blob, MAX_BLOB_BYTES) {
                if raw.first() == Some(&0x80) {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Decode one zlib-compressed `.npy` / `COO1` blob.
    pub fn decode_blob(&self, blob: &[u8], total_voxels: usize) -> CoreResult<DecodedBlob> {
        decode_blob_for_db(blob, total_voxels, &self.path_display())
    }

    /// List harvest episode IDs (Harvest schemas only).
    pub fn episode_ids(&self, limit: u32, offset: u32) -> CoreResult<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT episode_id FROM episodes ORDER BY created_at DESC, episode_id LIMIT ?1 OFFSET ?2",
        )?;
        let rows = stmt.query_map([limit as i64, offset as i64], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// List self-play game IDs.
    pub fn game_ids(&self, limit: u32, offset: u32) -> CoreResult<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT game_id FROM games ORDER BY created_at DESC, game_id LIMIT ?1 OFFSET ?2",
        )?;
        let rows = stmt.query_map([limit as i64, offset as i64], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Fetch a game-level row with optional columns replaced by NULL.
    pub fn game_row(&self, game_id: &str) -> CoreResult<Option<GameRow>> {
        let wanted = [
            "bc_masks_blob",
            "forces_blob",
            "game_id",
            "neural_engine",
            "checkpoint_version",
            "load_config",
            "bc_type",
            "resolution",
            "final_score",
            "final_compliance",
            "final_volume",
            "total_steps",
            "created_at",
            "initial_cantilever_problem_real_scale_factor",
            "stiffness_E",
        ];
        let select = self.select_with_nulls("games", &wanted)?;
        let sql = format!("SELECT {select} FROM games WHERE game_id = ?1");
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query([game_id])?;
        let Some(row) = rows.next()? else {
            return Ok(None);
        };
        Ok(Some(GameRow {
            bc_masks_blob: row.get(0)?,
            forces_blob: row.get(1)?,
            game_id: row.get(2)?,
            neural_engine: row.get(3)?,
            checkpoint_version: row.get(4)?,
            load_config: row.get(5)?,
            bc_type: row.get(6)?,
            resolution: row.get(7)?,
            final_score: row.get(8)?,
            final_compliance: row.get(9)?,
            final_volume: row.get(10)?,
            total_steps: row.get(11)?,
            created_at: row.get(12)?,
            real_scale_factor: row.get(13)?,
            stiffness_e: row.get(14)?,
        }))
    }
}

/// Dynamically typed SQL value used by the dataset command layer.
#[derive(Debug, Clone, PartialEq)]
pub enum SqlValue {
    Null,
    Integer(i64),
    Real(f64),
    Text(String),
    Blob(Vec<u8>),
}

impl SqlValue {
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            SqlValue::Integer(v) => Some(*v),
            SqlValue::Real(v) => Some(*v as i64),
            SqlValue::Text(v) => v.parse().ok(),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            SqlValue::Real(v) => Some(*v),
            SqlValue::Integer(v) => Some(*v as f64),
            SqlValue::Text(v) => v.parse().ok(),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            SqlValue::Text(v) => Some(v.as_str()),
            _ => None,
        }
    }

    pub fn as_blob(&self) -> Option<&[u8]> {
        match self {
            SqlValue::Blob(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, SqlValue::Null)
    }
}

impl ToSql for SqlValue {
    fn to_sql(&self) -> rusqlite::Result<rusqlite::types::ToSqlOutput<'_>> {
        use rusqlite::types::ToSqlOutput;
        match self {
            SqlValue::Null => Ok(ToSqlOutput::Borrowed(ValueRef::Null)),
            SqlValue::Integer(value) => Ok(ToSqlOutput::Borrowed(ValueRef::Integer(*value))),
            SqlValue::Real(value) => Ok(ToSqlOutput::Borrowed(ValueRef::Real(*value))),
            SqlValue::Text(value) => Ok(ToSqlOutput::Borrowed(ValueRef::Text(value.as_bytes()))),
            SqlValue::Blob(value) => Ok(ToSqlOutput::Borrowed(ValueRef::Blob(value))),
        }
    }
}

fn sql_value_from_ref(value: ValueRef<'_>) -> SqlValue {
    match value {
        ValueRef::Null => SqlValue::Null,
        ValueRef::Integer(v) => SqlValue::Integer(v),
        ValueRef::Real(v) => SqlValue::Real(v),
        ValueRef::Text(v) => SqlValue::Text(String::from_utf8_lossy(v).to_string()),
        ValueRef::Blob(v) => SqlValue::Blob(v.to_vec()),
    }
}

impl SqliteReader {
    /// Run a read-only dynamic query and materialize rows as snake_case maps.
    pub fn query_rows(
        &self,
        sql: &str,
        params: &[SqlValue],
    ) -> CoreResult<Vec<HashMap<String, SqlValue>>> {
        let mut stmt = self.conn.prepare(sql)?;
        let columns: Vec<String> = stmt
            .column_names()
            .iter()
            .map(|name| name.to_string())
            .collect();
        let mut rows = stmt.query(rusqlite::params_from_iter(params.iter()))?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            let mut map = HashMap::new();
            for (idx, name) in columns.iter().enumerate() {
                map.insert(name.clone(), sql_value_from_ref(row.get_ref(idx)?));
            }
            out.push(map);
        }
        Ok(out)
    }

    pub fn query_row(
        &self,
        sql: &str,
        params: &[SqlValue],
    ) -> CoreResult<Option<HashMap<String, SqlValue>>> {
        Ok(self.query_rows(sql, params)?.into_iter().next())
    }
}

/// Game metadata with optional legacy columns safely NULLed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameRow {
    #[serde(skip_serializing)]
    pub bc_masks_blob: Option<Vec<u8>>,
    #[serde(skip_serializing)]
    pub forces_blob: Option<Vec<u8>>,
    pub game_id: String,
    pub neural_engine: Option<String>,
    pub checkpoint_version: Option<String>,
    pub load_config: Option<String>,
    pub bc_type: Option<String>,
    pub resolution: Option<String>,
    pub final_score: Option<f64>,
    pub final_compliance: Option<f64>,
    pub final_volume: Option<f64>,
    pub total_steps: Option<i64>,
    pub created_at: Option<String>,
    pub real_scale_factor: Option<f64>,
    pub stiffness_e: Option<f64>,
}

fn validate_identifier(name: &str) -> CoreResult<()> {
    if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(CoreError::InvalidInput {
            message: format!("invalid SQL identifier: {name}"),
        });
    }
    Ok(())
}

/// Decode a compressed blob, surfacing legacy pickles explicitly.
pub fn decode_blob_for_db(
    blob: &[u8],
    total_voxels: usize,
    db_path: &str,
) -> CoreResult<DecodedBlob> {
    let raw = zlib_decompress(blob, MAX_BLOB_BYTES)?;
    if raw.starts_with(b"\x93NUMPY") {
        return Ok(DecodedBlob::Dense(parse_npy(&raw)?));
    }
    if raw.starts_with(b"COO1") {
        return Ok(DecodedBlob::Sparse(Coo1::parse(&raw, total_voxels)?));
    }
    if raw.first() == Some(&0x80) {
        return Err(CoreError::LegacyPickleBlob {
            db_path: db_path.to_string(),
        });
    }
    Err(CoreError::InvalidInput {
        message: format!(
            "unrecognized blob prefix in {db_path}: {:02x?}",
            &raw[..raw.len().min(8)]
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codecs::npy::encode_npy;
    use crate::codecs::zlib_compress;

    fn write_pickle_blob() -> Vec<u8> {
        let pickle = b"\x80\x02}q\x00.".to_vec();
        zlib_compress(&pickle).unwrap()
    }

    fn build_legacy_selfplay_db(path: &Path) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let conn = Connection::open(path).unwrap();
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                neural_engine TEXT NOT NULL,
                checkpoint_version TEXT NOT NULL,
                bc_masks_blob BLOB NOT NULL,
                forces_blob BLOB NOT NULL,
                load_config TEXT NOT NULL,
                bc_type TEXT NOT NULL,
                resolution TEXT NOT NULL,
                final_score REAL,
                final_compliance REAL,
                final_volume REAL,
                total_steps INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS game_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                phase TEXT NOT NULL,
                density_blob BLOB NOT NULL,
                policy_add_blob BLOB,
                policy_remove_blob BLOB,
                mcts_visit_add_blob BLOB,
                mcts_visit_remove_blob BLOB,
                mcts_q_add_blob BLOB,
                mcts_q_remove_blob BLOB,
                selected_actions_json TEXT,
                value REAL NOT NULL,
                mcts_stats_json TEXT,
                displacement_map_blob BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            "#,
        )
        .unwrap();
        let blob = write_pickle_blob();
        conn.execute(
            "INSERT OR REPLACE INTO games (game_id, neural_engine, checkpoint_version, bc_masks_blob, forces_blob, load_config, bc_type, resolution)
             VALUES ('legacy-game', 'simple', 'v2', X'00', X'00', '{}', 'FULL_CLAMP', '[8, 4, 2]')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO game_steps (game_id, step, phase, density_blob, value) VALUES ('legacy-game', 0, 'GROWTH', ?1, 0.0)",
            [blob],
        )
        .unwrap();
        drop(conn);
    }

    fn temp_selfplay_fixture(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "abcore-{tag}-{}-{}-selfplay_games.db",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn reads_real_or_fixture_selfplay_db_without_missing_columns() {
        let path = temp_selfplay_fixture("legacy-cols");
        build_legacy_selfplay_db(&path);
        let reader = SqliteReader::open(&path).unwrap();
        assert_eq!(reader.detect_kind().unwrap(), DbKind::SelfPlay);
        let tables = reader.tables().unwrap();
        assert!(tables.contains(&"games".to_string()));
        let columns = reader.columns("games").unwrap();
        assert!(columns.contains(&"game_id".to_string()));
        // A column absent from the legacy schema must be replaced by NULL, not fail.
        let select = reader
            .select_with_nulls(
                "games",
                &[
                    "game_id",
                    "fem_reward",
                    "displacement_map_blob",
                    "initial_cantilever_problem_real_scale_factor",
                    "stiffness_E",
                ],
            )
            .unwrap();
        assert!(select.contains("NULL AS fem_reward"));
        let sql = format!("SELECT {select} FROM games LIMIT 1");
        reader
            .connection()
            .query_row(&sql, [], |row| {
                assert_eq!(row.get::<_, String>(0).unwrap(), "legacy-game");
                let missing: Option<f64> = row.get(1).unwrap();
                assert!(missing.is_none());
                Ok(())
            })
            .unwrap();
        assert_eq!(reader.item_count(DbKind::SelfPlay).unwrap(), 1);
        assert_eq!(reader.total_steps(DbKind::SelfPlay).unwrap(), 1);
        let row = reader.game_row("legacy-game").unwrap().unwrap();
        assert_eq!(row.game_id, "legacy-game");
        assert!(row.stiffness_e.is_none());
        assert_eq!(
            reader.game_ids(10, 0).unwrap(),
            vec!["legacy-game".to_string()]
        );
        drop(reader);
        let _ = std::fs::remove_file(&path);

        // If a real data/selfplay_games.db is present, verify read-only introspection works too.
        let repo_db = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/selfplay_games.db");
        if repo_db.exists() {
            let real_reader = SqliteReader::open(&repo_db).unwrap();
            assert_eq!(real_reader.detect_kind().unwrap(), DbKind::SelfPlay);
            let real_select = real_reader
                .select_with_nulls(
                    "games",
                    &[
                        "game_id",
                        "fem_reward",
                        "displacement_map_blob",
                        "initial_cantilever_problem_real_scale_factor",
                        "stiffness_E",
                    ],
                )
                .unwrap();
            let real_sql = format!("SELECT {real_select} FROM games LIMIT 1");
            let mut stmt = real_reader.connection().prepare(&real_sql).unwrap();
            let _ = stmt.query_map([], |row| row.get::<_, String>(0)).unwrap();
        }
    }

    #[test]
    fn legacy_pickle_blob_is_reported_without_panic() {
        let path = temp_selfplay_fixture("legacy-pickle");
        build_legacy_selfplay_db(&path);
        let reader = SqliteReader::open(&path).unwrap();
        assert!(reader.detect_legacy_pickle(DbKind::SelfPlay).unwrap());
        let blob: Vec<u8> = reader
            .connection()
            .query_row(
                "SELECT density_blob FROM game_steps WHERE game_id='legacy-game' LIMIT 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        match reader.decode_blob(&blob, 64) {
            Err(CoreError::LegacyPickleBlob { db_path }) => {
                assert!(db_path.contains("selfplay_games.db"));
            }
            other => panic!("expected LegacyPickleBlob, got {other:?}"),
        }
        drop(reader);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn decodes_compressed_npy_and_coo1() {
        let tmp = std::env::temp_dir().join(format!(
            "abcore-codec-{}-{}.db",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        {
            let conn = Connection::open(&tmp).unwrap();
            conn.execute_batch("CREATE TABLE t (blob BLOB);").unwrap();
            let npy = encode_npy(
                &[1.0, 0.0, 1.0, 0.5],
                &[2, 2, 1],
                crate::codecs::npy::NpyDtype::F32,
            )
            .unwrap();
            let compressed = zlib_compress(&npy).unwrap();
            let mut stmt = conn.prepare("INSERT INTO t (blob) VALUES (?1)").unwrap();
            stmt.execute([compressed]).unwrap();
            drop(stmt);
            let coo = Coo1 {
                indices: vec![0, 2],
                values: vec![0.5, 1.0],
            }
            .encode(4)
            .unwrap();
            let compressed = zlib_compress(&coo).unwrap();
            let mut stmt = conn.prepare("INSERT INTO t (blob) VALUES (?1)").unwrap();
            stmt.execute([compressed]).unwrap();
            drop(stmt);
            let mut stmt = conn
                .prepare("INSERT INTO t (blob) VALUES (X'DEADBEEF')")
                .unwrap();
            stmt.execute([]).unwrap();
            drop(stmt);
        }
        let reader = SqliteReader::open(&tmp).unwrap();
        let mut stmt = reader
            .connection()
            .prepare("SELECT blob FROM t ORDER BY rowid")
            .unwrap();
        let blobs: Vec<Vec<u8>> = stmt
            .query_map([], |r| r.get(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(blobs.len(), 3);
        match reader.decode_blob(&blobs[0], 4).unwrap() {
            DecodedBlob::Dense(arr) => {
                assert_eq!(arr.shape, vec![2, 2, 1]);
                assert_eq!(arr.data[3], 0.5);
            }
            _ => panic!("expected dense"),
        }
        match reader.decode_blob(&blobs[1], 4).unwrap() {
            DecodedBlob::Sparse(coo) => {
                assert_eq!(coo.indices, vec![0, 2]);
                assert_eq!(coo.values, vec![0.5, 1.0]);
            }
            _ => panic!("expected sparse"),
        }
        assert!(reader.decode_blob(&blobs[2], 4).is_err());
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn open_missing_database_is_clean_error() {
        let err = SqliteReader::open("/tmp/abcore-does-not-exist.db");
        assert!(matches!(err, Err(CoreError::NotFound { .. })));
    }

    #[test]
    fn db_kind_serde_matches_as_str() {
        for kind in [
            DbKind::HarvestV33,
            DbKind::HarvestV2,
            DbKind::SelfPlay,
            DbKind::LegacyV1,
        ] {
            let json = serde_json::to_string(&kind).unwrap();
            assert_eq!(json, format!("\"{}\"", kind.as_str()));
            let roundtrip: DbKind = serde_json::from_str(&json).unwrap();
            assert_eq!(roundtrip, kind);
        }
    }
}
