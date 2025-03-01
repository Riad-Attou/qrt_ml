import sys
import pkg_resources

# requirements.txt の内容をリストとして定義
required_packages = {
    "asttokens": "3.0.0",
    "colorama": "0.4.6",
    "comm": "0.2.2",
    "contourpy": "1.3.1",
    "cycler": "0.12.1",
    "debugpy": "1.8.12",
    "decorator": "5.1.1",
    "ecos": "2.0.14",
    "executing": "2.2.0",
    "fonttools": "4.55.8",
    "ipykernel": "6.29.5",
    "ipython": "8.32.0",
    "jedi": "0.19.2",
    "joblib": "1.4.2",
    "jupyter_client": "8.6.3",
    "jupyter_core": "5.7.2",
    "kiwisolver": "1.4.8",
    "lightgbm": "4.5.0",
    "matplotlib": "3.10.0",
    "matplotlib-inline": "0.1.7",
    "nest-asyncio": "1.6.0",
    "numexpr": "2.10.2",
    "numpy": "2.2.2",
    "osqp": "0.6.7.post3",
    "packaging": "24.2",
    "pandas": "2.2.3",
    "parso": "0.8.4",
    "pillow": "11.1.0",
    "platformdirs": "4.3.6",
    "prompt_toolkit": "3.0.50",
    "psutil": "6.1.1",
    "pure_eval": "0.2.3",
    "Pygments": "2.19.1",
    "pyparsing": "3.2.1",
    "python-dateutil": "2.9.0.post0",
    "pytz": "2025.1",
    "pywin32": "308",
    "pyzmq": "26.2.1",
    "qdldl": "0.1.7.post5",
    "scikit-learn": "1.5.2",
    "scikit-survival": "0.23.1",
    "scipy": "1.15.1",
    "six": "1.17.0",
    "stack-data": "0.6.3",
    "threadpoolctl": "3.5.0",
    "tornado": "6.4.2",
    "traitlets": "5.14.3",
    "typing_extensions": "4.12.2",
    "tzdata": "2025.1",
    "wcwidth": "0.2.13",
}

print("\n✅ 仮想環境の Python 実行パス:", sys.executable)

# インストールされているパッケージを取得
installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

# チェック処理
missing_packages = []
version_mismatches = []

for package, required_version in required_packages.items():
    if package not in installed_packages:
        missing_packages.append(package)
    elif installed_packages[package] != required_version:
        version_mismatches.append((package, installed_packages[package], required_version))

# 結果表示
if not missing_packages and not version_mismatches:
    print("✅ 仮想環境は正常にセットアップされています。")
else:
    if missing_packages:
        print("\n❌ 以下のパッケージが見つかりませんでした:")
        for pkg in missing_packages:
            print(f"  - {pkg}")

    if version_mismatches:
        print("\n⚠️ 以下のパッケージのバージョンが一致しません:")
        for pkg, installed, expected in version_mismatches:
            print(f"  - {pkg}: インストール済み {installed}, 期待値 {expected}")

print("\n🎯 チェック完了！")
