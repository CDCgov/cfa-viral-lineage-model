METADATA=metadata.csv

.PHONY: all

all: $(METADATA)
	python evaluate.py $<

out/samples-baseline.csv: fit_baseline_model.py $(METADATA)
	python $< > $@

out/samples-id.csv: fit_indep_div_model.py $(METADATA)
	python $< $(METADATA) > $@

$(METADATA): ../../data/load_metadata.py
	python $< > $@
