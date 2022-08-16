# HNGCL

Source code for Hard Negative Synthesis for Graph Contrastive Learning

For example, to run GCA-Degree under WikiCS, execute:

    python main_0.py --device cuda:0 --dataset WikiCS --param local:wikics.json
    --dataset Amazon-Computers --param local:amazon_computers.json
    --device cuda:2 --dataset Amazon-Photo --param local:amazon_photo.json
    --device cuda:3 --dataset Coauthor-CS --param local:coauthor_cs.json
    --device cuda:3 --dataset Coauthor-Phy --param local:coauthor_phy.json


