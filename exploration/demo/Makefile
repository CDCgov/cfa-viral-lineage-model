METADATA=metadata.csv

.PHONY: all

all: $(METADATA) out/samples-baseline.csv out/samples-id.csv
	python3 evaluate.py $<

out/samples-baseline.csv: fit_baseline_model.py $(METADATA)
	python3 $< $(METADATA) > $@

out/samples-id.csv: fit_indep_div_model.py $(METADATA)
	python3 $< $(METADATA) > $@

$(METADATA): ../../linmod/data.py
	python3 -m linmod.data > $@
