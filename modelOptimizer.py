import os
from trainConvNet import main as trainMain
from testConvNet import main as testMain
import optuna
from convNet import ConvNet
from hybridNet import ConvNet as hybridNet
from thinNet import thinNet
from lowKernelNet import lkNet

listOfNets = [ConvNet, hybridNet, thinNet, lkNet]
listOfNames = ["ConvNet", "hybridNet", "thinNet", "lkNet"]


def objective(trial):
    # Define the hyperparameters to optimize
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 1, 10)
    num_epochs = trial.suggest_int("num_epochs", 1, 50)
    noise = trial.suggest_float("noise", 0.0, 0.5)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    print("hyperparameters: ")
    print("learning_rate: ", learning_rate)
    print("weight_decay: ", weight_decay)
    print("batch_size: ", batch_size)
    print("num_epochs: ", num_epochs)
    print("noise: ", noise)
    print("dropout: ", dropout)

    # Call the training function with the hyperparameters
    trainAcc, testAcc = trainMain(netArch=netArch,
        nameTrain ="train_250",
        nameValid = "valid_250",
        checkpointName = checkpointFile,
        learning_rate=learning_rate, 
        weight_decay=weight_decay, 
        batch_size=batch_size, 
        num_epochs=num_epochs, 
        noise=noise, 
        dropout=dropout,
        )
    
    accResult = max(testAcc)
    
    # Remove the checkpoint file after testing
    os.remove("checkpoint/"+checkpointFile)
    
    with open("study/studyFULL"+str(netName)+".txt", "a") as f:
        f.write("\n")
        f.write("\n")
        f.write("\n")
        f.write("Trial number: " + str(trial.number) + "\n")
        f.write("Current trial for " + str(netName) + "\n")
        f.write(" Hyperparameters: \n")
        f.write("  learning_rate: {}\n".format(learning_rate))
        f.write("  weight_decay: {}\n".format(weight_decay))
        f.write("  batch_size: {}\n".format(batch_size))
        f.write("  total num_epochs: {}\n".format(num_epochs))
        f.write("  noise: {}\n".format(noise))
        f.write("  dropout: {}\n".format(dropout))
        f.write("  train accuracy list after each epoch: {}\n".format(trainAcc))
        f.write("  valid accuracy list after each epoch: {}\n".format(testAcc))
        f.write("  valid accuracy: {}\n".format(accResult))
        f.write("\n")
        f.write("\n")
        f.write("\n")

    return accResult

def stop100(study, trial):
    if trial.value >= 80:
        print("Stopping early because the accuracy is above 80")
        study.stop()

for netArch in listOfNets:
    netName = listOfNames[listOfNets.index(netArch)]
    checkpointFile = "checkpoint"+str(netName)+".pth"
    print("Optimizing for ", netName)
    # Create a study and optimize the objective function
    study = optuna.create_study(direction="maximize", storage="sqlite:///study/study"+str(netArch)+".db", load_if_exists=True)
    study.optimize(objective, n_trials=1000, callbacks=[stop100])

    with open("study/study"+str(netArch)+".txt", "a") as f:
        f.write("Best trial for " + str(netArch) + "\n")
        print("Best trial:")
        trial = study.best_trial
        f.write("  Value: {}\n".format(trial.value))
        f.write("  Params: \n")
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
            f.write("    {}: {}\n".format(key, value))
        f.write("\n")
        print("\n")
        print("\n")
