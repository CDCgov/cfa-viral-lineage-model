METADATA=metadata.csv
SAMPLE_FILES=out/samples-baseline.csv out/samples-id.csv

.PHONY: all clean

all: results.txt

results.txt: $(METADATA) $(SAMPLE_FILES)
	python3 evaluate.py $< > $@

out/samples-baseline.csv: fit_baseline_model.py $(METADATA)
	python3 $< $(METADATA) > $@

out/samples-id.csv: fit_indep_div_model.py $(METADATA)
	python3 $< $(METADATA) > $@

$(METADATA): ../../linmod/data.py
	python3 -m linmod.data > $@

clean:
	rm -f $(METADATA) $(SAMPLE_FILES) results.txt
	rm -rf .cache/*
