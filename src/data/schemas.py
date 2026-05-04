"""
Pydantic schemas for KDD99 data validation.
"""
from typing import Literal
from pydantic import BaseModel, Field, validator
import pandas as pd

class KDD99Row(BaseModel):
    """Single KDD99 connection record."""
    duration: int = Field(..., ge=0)
    protocol_type: Literal['tcp', 'udp', 'icmp']
    service: str = Field(..., max_length=20)
    flag: Literal['SF', 'S0', 'REJ', 'RSTR', 'S1', 'S2', 'RSTOS0', 'SH', 'RSTO', 'S3', 'OTH']
    src_bytes: int = Field(..., ge=0)
    dst_bytes: int = Field(..., ge=0)
    land: int = Field(..., ge=0, le=1)
    wrong_fragment: int = Field(..., ge=0)
    urgent: int = Field(..., ge=0)
    hot: int = Field(..., ge=0)
    num_failed_logins: int = Field(..., ge=0)
    logged_in: int = Field(..., ge=0, le=1)
    num_compromised: int = Field(..., ge=0)
    root_shell: int = Field(..., ge=0, le=1)
    su_attempted: int = Field(..., ge=0, le=1)
    num_root: int = Field(..., ge=0)
    num_file_creations: int = Field(..., ge=0)
    num_shells: int = Field(..., ge=0)
    num_access_files: int = Field(..., ge=0)
    is_host_login: int = Field(..., ge=0, le=1)
    is_guest_login: int = Field(..., ge=0, le=1)
    count: int = Field(..., ge=0)
    srv_count: int = Field(..., ge=0)
    serror_rate: float = Field(..., ge=0.0, le=1.0)
    srv_serror_rate: float = Field(..., ge=0.0, le=1.0)
    rerror_rate: float = Field(..., ge=0.0, le=1.0)
    srv_rerror_rate: float = Field(..., ge=0.0, le=1.0)
    same_srv_rate: float = Field(..., ge=0.0, le=1.0)
    diff_srv_rate: float = Field(..., ge=0.0, le=1.0)
    srv_diff_host_rate: float = Field(..., ge=0.0, le=1.0)
    dst_host_count: int = Field(..., ge=0)
    dst_host_srv_count: int = Field(..., ge=0)
    dst_host_same_srv_rate: float = Field(..., ge=0.0, le=1.0)
    dst_host_diff_srv_rate: float = Field(..., ge=0.0, le=1.0)
    dst_host_same_src_port_rate: float = Field(..., ge=0.0, le=1.0)
    dst_host_srv_diff_host_rate: float = Field(..., ge=0.0, le=1.0)
    dst_host_serror_rate: float = Field(..., ge=0.0, le=1.0)
    dst_host_srv_serror_rate: float = Field(..., ge=0.0, le=1.0)
    dst_host_rerror_rate: float = Field(..., ge=0.0, le=1.0)
    dst_host_srv_rerror_rate: float = Field(..., ge=0.0, le=1.0)
    connection_type: Literal['normal', 'dos', 'probe', 'r2l', 'u2r']

    class Config:
        arbitrary_types_allowed = True

    @validator('*', pre=True, always=True)
    def coerce_types(cls, v):
        if isinstance(v, str):
            return v.strip()
        return v

def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate DF rows with Pydantic."""
    validated_rows = []
    for _, row in df.iterrows():
        try:
            validated = KDD99Row(**row.to_dict())
            validated_rows.append(validated.dict())
        except ValueError as e:
            print(f"Validation error: {e}")
            continue
    return pd.DataFrame(validated_rows)
