## Custom Consensus Engine for go-ethereum

##### Create ethereum directory in your $GOPATH/src/github.com/
    mkdir ethereum

##### Clone repository in $GOPATH/src/github.com/ethereum/
    git clone https://github.com/shrikantgoswami/go-ethereum.git 
    
##### Build
    go build -o geth github.com/ethereum/go-ethereum/cmd/geth
    
##### Run 
    ./run-custom-consensus.sh
    
##### In a new terminal window attach to the console
    ./geth attach ~/.ethereum/xdposnet/geth.ipc
    
 ##### Create new account
    personal.newAccount()
    
 ##### Start miner
    miner.start(1)
 
 The output in the first terminal window should now emit mining details.