METADATA=metadata.csv

.PHONY: all

all: $(METADATA) out/samples-baseline.csv out/samples-id.csv
	python evaluate.py $<

out/samples-baseline.csv: fit_baseline_model.py $(METADATA)
	python $< $(METADATA) > $@

out/samples-id.csv: fit_indep_div_model.py $(METADATA)
	python $< $(METADATA) > $@

$(METADATA): ../../linmod/data.py
	python3 -m linmod.data > $@
