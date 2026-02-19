"""config 모듈 테스트"""


import pytest

from airgap_kor_search.config import AppConfig, load_or_create_config


class TestAppConfig:
    """AppConfig 기본 동작 테스트"""

    def test_default_values(self):
        config = AppConfig()
        assert config.model.embedding_dim == 1024
        assert config.chunk.chunk_size == 512
        assert config.search.top_k == 5
        assert config.server.port == 8007

    def test_ensure_dirs(self, tmp_path):
        config = AppConfig(
            data_dir=tmp_path / "test_data",
            model={"model_dir": tmp_path / "test_data" / "model"},
        ) if False else AppConfig()
        # 간단하게 기본 config로 테스트
        config.data_dir = tmp_path / "test_data"
        config.ensure_dirs()
        assert config.data_dir.exists()

    def test_to_dict(self):
        config = AppConfig()
        data = config.to_dict()
        assert isinstance(data, dict)
        assert isinstance(data["data_dir"], str)
        assert "model" in data
        assert "chunk" in data

    def test_save_and_load(self, tmp_path):
        config = AppConfig()
        config_path = tmp_path / "config.json"
        config.save(config_path)

        assert config_path.exists()

        loaded = AppConfig.load(config_path)
        assert loaded.model.embedding_dim == config.model.embedding_dim
        assert loaded.chunk.chunk_size == config.chunk.chunk_size
        assert loaded.search.top_k == config.search.top_k

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AppConfig.load(tmp_path / "nonexistent.json")

    def test_from_dict(self):
        data = {
            "data_dir": "/custom/path",
            "model": {"embedding_dim": 768},
            "chunk": {"chunk_size": 256},
            "search": {"top_k": 10},
            "server": {"port": 9000},
        }
        config = AppConfig.from_dict(data)
        assert config.model.embedding_dim == 768
        assert config.chunk.chunk_size == 256
        assert config.search.top_k == 10
        assert config.server.port == 9000


class TestLoadOrCreateConfig:
    """load_or_create_config 함수 테스트"""

    def test_creates_default_when_missing(self, tmp_path):
        config_path = tmp_path / "new_config.json"
        config = load_or_create_config(config_path)

        assert config_path.exists()
        assert isinstance(config, AppConfig)

    def test_laods_existing(self, tmp_path):
        config_path = tmp_path / "config.json"
        original = AppConfig()
        original.save(config_path)

        loaded = load_or_create_config(config_path)
        assert loaded.model.embedding_dim == original.model.embedding_dim
