METADATA=metadata.csv
FORECAST_DIR=out/

.PHONY: all clean

all: results.txt

results.txt: $(METADATA)
	python3 main.py $< > $@

$(METADATA): ../../linmod/data.py
	python3 -m linmod.data > $@

clean:
	rm -f $(METADATA) results.txt
	rm -rf .cache/* $(FORECAST_DIR)
