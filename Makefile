IP = 
USER = ubuntu
PORT = 22
ID_FILE = 
PRJ_NAME = 
SERVER_PROJECT_PATH = /home/$(USER)/$(PRJ_NAME)
PATH_MY_OS = 

deps:
	sudo apt install make

run:
	python3 avl_optmizer.py

ssh:
	ssh -i $(ID_FILE) $(USER)@$(IP)

ssh_nopass:
	ssh $(USER)@$(IP) -p $(PORT) 

install:
	rsync -av --exclude=".git" -e 'ssh -i $(ID_FILE) -p $(PORT)' . $(USER)@$(IP):$(SERVER_PROJECT_PATH)
	@echo "app installed on target:$(SERVER_PROJECT_PATH)"

pull:
	rsync -av --exclude=".git" -e 'ssh -i $(ID_FILE) -p $(PORT)' $(USER)@$(IP):$(SERVER_PROJECT_PATH) .. 
	@echo "app installed on target:$(SERVER_PROJECT_PATH)"


parse_util:
	python3 avl_parse_util.py

clean:
	find env -name "out_*.txt" -exec rm -f {} \;
	find env -name "in_*.avl" -exec rm -f {} \;

commita:
	@git add .
	@git commit -m "$(shell date)"
	@git push origin master

.PHONY: all