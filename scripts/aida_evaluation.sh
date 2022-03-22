echo "### Evaluating on Aida Validation Dataset ###"
classy evaluate $1 "data/aida/validation.aida" -pd extend
echo "### Evaluating on Aida Test Dataset ###"
classy evaluate $1 "data/aida/test.aida" -pd extend
