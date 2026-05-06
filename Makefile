.PHONY: help serve test test-js test-py fixtures build-preset inline-preset bundle clean

help:
	@echo "Available targets:"
	@echo "  make serve          - Start local server at http://localhost:8000/"
	@echo "  make test           - Run all tests (JS + Python)"
	@echo "  make test-js        - Run JS unit tests (node --test)"
	@echo "  make test-py        - Run Python tests (pytest, numpy reference)"
	@echo "  make fixtures       - Regenerate tests/fixtures/*.json from numpy reference"
	@echo "  make build-preset   - Build hand-crafted preset → presets/japanese-mini-v1.json"
	@echo "  make inline-preset  - Embed presets/japanese-mini-v1.json into src/js/presets.js"
	@echo "  make bundle         - Build single-HTML deliverable to build/nn_sim_transformer.html"
	@echo "  make clean          - Remove build artifacts"

serve:
	cd src && python3 -m http.server 8000

test: test-js test-py

test-js:
	@if ls tests/js/*.test.mjs >/dev/null 2>&1; then \
		node --test tests/js/*.test.mjs; \
	else \
		echo "No JS tests yet (tests/js/*.test.mjs)"; \
	fi

test-py:
	uv run pytest tests/py/ -q -p no:cacheprovider

fixtures:
	uv run python tests/py/generate_fixtures.py

# Hand-crafted preset を構築して JSON 出力する (P2 で実装)。
# 設計書 §6.9 の表に従って W_E / W_Q / W_K / W_V / W_O / FFN を直接書き込む。
# 学習はしない (= autograd / loss 監視 / epoch ループは無し)。
# シミュレータ本体には含まれない開発者向けスクリプト。
build-preset:
	uv run python tools/build_preset.py

# 生成された preset を src/js/presets.js に inline する (P3 で実装)。
# bundle 時にバンドル内に埋め込まれて、起動時の fetch が不要になる。
inline-preset:
	uv run python tools/inline_preset.py

bundle:
	uv run python tools/bundle.py

clean:
	rm -rf build/nn_sim_transformer.html
	rm -rf .pytest_cache __pycache__
