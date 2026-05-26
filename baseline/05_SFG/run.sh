DONE_FILE="done_commands.txt"

if [ ! -f "$DONE_FILE" ]; then
    touch "$DONE_FILE"
fi

mapfile -t done_commands < "$DONE_FILE"

commands=(
	# german SAGE
	"python sfg.py --dataset='german' --encoder='SAGE' --d_epochs=5 --g_epochs=10 --c_epochs=10 --c_lr=0.001 --e_lr=0.01 --ratio=0.5"      # FairVGNN
	"python sfg.py --dataset='german' --encoder='SAGE' --d_epochs=5 --g_epochs=10 --c_epochs=10 --c_lr=0.001 --e_lr=0.01 --ratio=0.5 --with_constraint --rho=2"
	"python sfg.py --dataset='german' --encoder='SAGE' --d_epochs=5 --g_epochs=10 --c_epochs=10 --c_lr=0.001 --e_lr=0.01 --ratio=0.5 --with_constraint --rho=2 --loss_alpha=0.5"   # SFG
	
	# bail SAGE
	"python sfg.py --dataset='bail' --encoder='SAGE' --d_epochs=5 --g_epochs=10 --c_epochs=5 --c_lr=0.01 --e_lr=0.001 --ratio=1 --epochs=160"
    "python sfg.py --dataset='bail' --encoder='SAGE' --d_epochs=5 --g_epochs=10 --c_epochs=5 --c_lr=0.01 --e_lr=0.001 --ratio=1 --epochs=160 --with_constraint --rho=2"  
    "python sfg.py --dataset='bail' --encoder='SAGE' --d_epochs=5 --g_epochs=10 --c_epochs=5 --c_lr=0.01 --e_lr=0.001 --ratio=1 --epochs=160 --with_constraint --rho=2 --loss_alpha=0.5"   
	    
	# # credit SAGE
    "python sfg_credit.py --dataset='credit' --encoder='SAGE' --d_epochs=5 --g_epochs=5 --c_epochs=10 --c_lr=0.01 --e_lr=0.001 --ratio=0"
	"python sfg_credit.py --dataset='credit' --encoder='SAGE' --d_epochs=5 --g_epochs=5 --c_epochs=10 --c_lr=0.01 --e_lr=0.001 --ratio=0 --with_constraint --rho=2"
	"python sfg_credit.py --dataset='credit' --encoder='SAGE' --d_epochs=5 --g_epochs=5 --c_epochs=10 --c_lr=0.01 --e_lr=0.001 --ratio=0 --with_constraint --rho=2 --loss_alpha=0.5"
  )

for cmd in "${commands[@]}"; do
    if [[ ! " ${done_commands[*]} " =~ " ${cmd} " ]]; then
        echo "Executing: $cmd"
        eval $cmd
        echo "$cmd" >> "$DONE_FILE"
    else
        echo "Skipping: $cmd (already done)"
    fi
done

echo "All commands completed."