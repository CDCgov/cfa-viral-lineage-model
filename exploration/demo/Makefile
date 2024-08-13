METADATA=metadata.csv
FORECAST_DIR=out/

.PHONY: all clean

all: results.csv

results.csv: $(METADATA)
	python3 main.py $< > $@

$(METADATA): ../../linmod/data.py
	python3 -m linmod.data > $@

clean:
	rm -f $(METADATA) results.csv
	rm -rf .cache/* $(FORECAST_DIR)
