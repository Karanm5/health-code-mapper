.PHONY: setup data etl map test app all clean

setup:          ## Install dependencies
	pip install -r requirements.txt

data:           ## Build the demo vocabulary and synthetic UKB-shaped source data
	python data/build_vocabulary.py
	python data/generate_source.py

etl:            ## Run the dbt ETL: source -> OMOP CDM (DuckDB)
	mkdir -p outputs
	cd dbt && dbt run --profiles-dir .

map:            ## Run the conformal mapping pipeline and write outputs/
	python scripts/run_mapping.py

test:           ## Run the test suite
	python -m pytest -q

app:            ## Launch the interactive demo
	streamlit run app/streamlit_app.py

all: data etl map test   ## Build everything end to end

clean:          ## Remove generated artefacts
	rm -rf outputs data/source data/vocab dbt/target dbt/logs
	find . -name __pycache__ -type d -exec rm -rf {} +
