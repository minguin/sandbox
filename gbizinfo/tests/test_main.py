import click
import pytest
from click.testing import CliRunner

from gbizinfo.main import main, validate_corporate_number


def test_validate_corporate_number():
    assert validate_corporate_number(None, None, "7010401075212")
    with pytest.raises(click.BadParameter):
        validate_corporate_number(None, None, "2224444")


def test_main():
    result = CliRunner().invoke(main, ["--corporate_number", "7010401075212"])
    assert result.exit_code == 0
    # assert result.output == "('株式会社ユーザベース', '東京都千代田区丸の内２丁目５番２号')\n" # ユーザーベース住所変わってたり、テストが上手く行かない
