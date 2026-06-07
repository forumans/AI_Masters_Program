from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    api_key: str
    host: str = "0.0.0.0"
    port: int = 8000
    db_min_connections: int = 2
    db_max_connections: int = 10

    model_config = {"env_file": ".env"}


settings = Settings()
