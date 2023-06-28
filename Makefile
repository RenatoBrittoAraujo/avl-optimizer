run:
	python3 avl_optmizer.py

parse_util:
	python3 avl_parse_util.py

clean:
	find env -name "out_*.txt" -exec rm -f {} \;
	find env -name "in_*.avl" -exec rm -f {} \;
