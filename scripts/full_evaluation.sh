echo "##### IN-DOMAIN #####"
echo "### Evaluating on Aida Validation Dataset ###"
classy evaluate $1 "data/aida/validation.aida" -pd extend
echo "### Evaluating on Aida Test Dataset ###"
classy evaluate $1 "data/aida/test.aida" -pd extend

echo "##### OUT-OF-DOMAIN #####"
echo "### Evaluating on MSNBC Dataset ###"
classy evaluate $1 "data/out_of_domain/msnbc-test-kilt.ed" -pd extend
echo "### Evaluating on AQUAINT Dataset ###"
classy evaluate $1 "data/out_of_domain/aquaint-test-kilt.ed" -pd extend
echo "### Evaluating on ACE2004 Dataset ###"
classy evaluate $1 "data/out_of_domain/ace2004-test-kilt.ed" -pd extend
echo "### Evaluating on CWEB Dataset ###"
classy evaluate $1 "data/out_of_domain/clueweb-test-kilt.ed" -pd extend
echo "### Evaluating on WIKI Dataset ###"
classy evaluate $1 "data/out_of_domain/wiki-test-kilt.ed" -pd extend
