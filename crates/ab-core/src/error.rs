//! Stable error type shared by every Rust-facing subsystem.

use serde::Serialize;

/// Result alias used across `ab-core`.
pub type CoreResult<T> = Result<T, CoreError>;

/// Errors with stable machine-readable codes.
#[derive(Debug, Clone, thiserror::Error)]
pub enum CoreError {
    #[error("invalid input: {message}")]
    InvalidInput { message: String },

    #[error("resolution limit exceeded: {message}")]
    ResolutionLimitExceeded { message: String },

    #[error("path escapes the configured sandbox: {path}")]
    PathOutsideSandbox { path: String },

    #[error("not found: {what}")]
    NotFound { what: String },

    #[error("unsupported operation: {message}")]
    Unsupported { message: String },

    #[error("legacy pickle blob detected in {db_path}")]
    LegacyPickleBlob { db_path: String },

    #[error("invalid .npy payload: {message}")]
    Npy { message: String },

    #[error("unsupported .npy dtype: {dtype}")]
    UnsupportedDtype { dtype: String },

    #[error(".npy payload uses fortran_order=true, which is not supported")]
    FortranOrder,

    #[error("invalid COO1 payload: {message}")]
    Coo1 { message: String },

    #[error("invalid ABWIRE01 payload: {message}")]
    Wire { message: String },

    #[error("invalid ABMESH01 payload: {message}")]
    AbMesh { message: String },

    #[error("SQLite error: {message}")]
    Sqlite { message: String },

    #[error("I/O error: {message}")]
    Io { message: String },

    #[error("JSON error: {message}")]
    Json { message: String },
}

impl CoreError {
    /// Stable error code (identical to the Python engine envelope).
    pub fn code(&self) -> &'static str {
        match self {
            CoreError::InvalidInput { .. } => "INVALID_INPUT",
            CoreError::ResolutionLimitExceeded { .. } => "RESOLUTION_LIMIT_EXCEEDED",
            CoreError::PathOutsideSandbox { .. } => "PATH_OUTSIDE_SANDBOX",
            CoreError::NotFound { .. } => "NOT_FOUND",
            CoreError::Unsupported { .. } => "CAPABILITY_UNAVAILABLE",
            CoreError::LegacyPickleBlob { .. } => "LEGACY_PICKLE_BLOB",
            CoreError::Npy { .. } => "INVALID_NPY",
            CoreError::UnsupportedDtype { .. } => "UNSUPPORTED_DTYPE",
            CoreError::FortranOrder => "FORTRAN_ORDER_UNSUPPORTED",
            CoreError::Coo1 { .. } => "INVALID_COO1",
            CoreError::Wire { .. } => "INVALID_ABWIRE01",
            CoreError::AbMesh { .. } => "INVALID_ABMESH01",
            CoreError::Sqlite { .. } => "SQLITE_ERROR",
            CoreError::Io { .. } => "IO_ERROR",
            CoreError::Json { .. } => "INTERNAL_ERROR",
        }
    }

    /// Additional structured fields attached to the envelope.
    pub fn details(&self) -> serde_json::Value {
        match self {
            CoreError::LegacyPickleBlob { db_path } => {
                serde_json::json!({ "db_path": db_path })
            }
            CoreError::PathOutsideSandbox { path } => serde_json::json!({ "path": path }),
            CoreError::NotFound { what } => serde_json::json!({ "what": what }),
            _ => serde_json::Value::Null,
        }
    }

    /// Standard error envelope (snake_case), as consumed by the Studio UI.
    pub fn to_envelope(&self) -> ErrorEnvelope {
        ErrorEnvelope {
            error: ErrorBody {
                code: self.code().to_string(),
                message: self.to_string(),
                details: self.details(),
                request_id: None,
            },
        }
    }
}

impl From<rusqlite::Error> for CoreError {
    fn from(value: rusqlite::Error) -> Self {
        CoreError::Sqlite {
            message: value.to_string(),
        }
    }
}

impl From<std::io::Error> for CoreError {
    fn from(value: std::io::Error) -> Self {
        CoreError::Io {
            message: value.to_string(),
        }
    }
}

impl From<serde_json::Error> for CoreError {
    fn from(value: serde_json::Error) -> Self {
        CoreError::Json {
            message: value.to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorEnvelope {
    pub error: ErrorBody,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorBody {
    pub code: String,
    pub message: String,
    pub details: serde_json::Value,
    pub request_id: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_is_snake_case_and_stable() {
        let err = CoreError::LegacyPickleBlob {
            db_path: "data/selfplay_games.db".into(),
        };
        let envelope = err.to_envelope();
        assert_eq!(envelope.error.code, "LEGACY_PICKLE_BLOB");
        assert_eq!(envelope.error.details["db_path"], "data/selfplay_games.db");
        let json = serde_json::to_value(&envelope).unwrap();
        assert_eq!(json["error"]["code"], "LEGACY_PICKLE_BLOB");
        assert!(json["error"].get("request_id").is_some());
    }
}
