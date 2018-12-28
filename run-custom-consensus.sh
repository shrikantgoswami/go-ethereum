#!/bin/bash

# Clear any ethereum stuff that exists
rm -rf ~/.ethereum/xdposnet

# Initalize with custom genesis file
./geth --datadir ~/.ethereum/xdposnet init xdposgenesis.json

# Launch geth
./geth -rpc -rpcapi 'web3,eth,debug,personal' -rpcport 8545 --rpccorsdomain '*' --datadir ~/.ethereum/xdposnet --networkid 99