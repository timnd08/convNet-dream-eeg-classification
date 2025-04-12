from convNet import ConvNet
from hybridNet import ConvNet as hybridNet
from thinNet import thinNet
from lowKernelNet import lkNet
from trainConvNet import main as trainMain
from testConvNet import main as testMain
import os
import sys

listOfNets = [ConvNet, hybridNet, thinNet, lkNet]
listOfNames = ["ConvNet", "hybridNet","thinNet", "lkNet"]

listOfCheckpoints = ["checkpointConvNet.pth", 
                    "checkpointHybridNet.pth",
                    "checkpointThinNet.pth", 
                    "checkpointLowKernelNet.pth"]

listOfLearningRates = [
  2.3268893476218194e-05,
    5.952683874842878e-05,
4.9000431533600275e-05,
0.009794120021934416
]
listOfWeightDecays = [
    0.002037387963490258,
    7.923343806367472e-05,
    0.00520772482765386,
    0.00012214065367421588
]

listOfBatchSizes = [6,3,7,3]

listOfNumEpochs = [6,7,37,7]

listOfNoise = [0.05973972259178478,
0.4509375570767015,
0.38750371021501107, 
0.08211924166420309]

listOfDropout = [0.13647238825089068, 0.3688525262501296, 0.3498703513828524, 0.08366519015123108]

result = "RESULT.txt"

inputSet = int(sys.argv[1])

with open(result, "a") as f:
    f.write("Fully optimized parameters, except that this one has more epochs\n")
    for i in range(inputSet-1,inputSet):
        print("Training net: ", listOfNames[i])
        f.write("Training net: " + listOfNames[i] + "\n")

        netArch = listOfNets[i]
        checkpointFile = "test.pth"

        learning_rate = listOfLearningRates[i]
        weight_decay = listOfWeightDecays[i]
        batch_size = listOfBatchSizes[i]
        num_epochs = 50
        noise = listOfNoise[i]
        dropout = listOfDropout[i]
        print("Hyperparameters: ")
        print("Learning rate: ", learning_rate)
        print("Weight decay: ", weight_decay)
        print("Batch size: ", batch_size)
        print("Num epochs: ", num_epochs)
        print("Noise: ", noise)
        print("Dropout: ", dropout)

        # Call the training function with the hyperparameters
        trainMain(netArch=netArch,
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
        
        accResultValid = testMain(
            netArch=netArch,
            name = "valid_250",
            checkpointName = checkpointFile
            )

        print("Validation accuracy: ", accResultValid)
        f.write("Validation accuracy: " + str(accResultValid) + "\n")

        accResultTest = testMain(
            netArch=netArch,
            name = "test_250",
            checkpointName = checkpointFile
            )
        
        print("Test accuracy: ", accResultTest)
        print("\n\n\n")
        f.write("Test accuracy: " + str(accResultTest) + "\n")

        os.remove("checkpoint/"+checkpointFile)
    
    f.write("\n\n\n")
    f.write("====================================================\n")

    
    # f.write("\n\n\n")
    # f.write("====================================================\n")

   
    # for i in range(inputSet-1,inputSet):
    #     print("Training net: ", listOfNames[i])
    #     f.write("Training net: " + listOfNames[i] + "\n")

    #     netArch = listOfNets[i]
    #     checkpointFile = "test.pth"
    #     learning_rate = 0.001
    #     weight_decay = listOfWeightDecays[i]
    #     batch_size = listOfBatchSizes[i]
    #     num_epochs = 20
    #     noise = listOfNoise[i]
    #     dropout = listOfDropout[i]

    #     print("Hyperparameters: ")
    #     print("Learning rate: ", learning_rate)
    #     print("Weight decay: ", weight_decay)
    #     print("Batch size: ", batch_size)
    #     print("Num epochs: ", num_epochs)
    #     print("Noise: ", noise)
    #     print("Dropout: ", dropout)

    #     # Call the training function with the hyperparameters
    #     trainMain(netArch=netArch,
    #         nameTrain ="train_250",
    #         nameValid = "valid_250",
    #         checkpointName = checkpointFile,
    #         learning_rate=learning_rate, 
    #         weight_decay=weight_decay, 
    #         batch_size=batch_size, 
    #         num_epochs=num_epochs, 
    #         noise=noise, 
    #         dropout=dropout,
    #         )
        
    #     accResultValid = testMain(
    #         netArch=netArch,
    #         name = "valid_250",
    #         checkpointName = checkpointFile
    #         )

    #     print("Validation accuracy: ", accResultValid)
    #     f.write("Validation accuracy: " + str(accResultValid) + "\n")

    #     accResultTest = testMain(
    #         netArch=netArch,
    #         name = "test_250",
    #         checkpointName = checkpointFile
    #         )
        
    #     print("Test accuracy: ", accResultTest)
    #     f.write("Test accuracy: " + str(accResultTest) + "\n")

    #     os.remove("checkpoint/"+checkpointFile)

    #     f.write("Optimize learning rate\n")
    
    
    f.write("\n\n\n")
    f.write("====================================================\n")


    #f.write("Optimized weightdecay\n")
    for i in range(inputSet-1,inputSet):
        print("Training net: ", listOfNames[i])
        f.write("Training net: " + listOfNames[i] + "\n")

        netArch = listOfNets[i]
        checkpointFile = "test.pth"
        learning_rate = listOfLearningRates[i]
        weight_decay = 0.0
        batch_size = listOfBatchSizes[i]
        num_epochs = 50
        noise = listOfNoise[i]
        dropout = listOfDropout[i]

        print("Hyperparameters: ")
        print("Learning rate: ", learning_rate)
        print("Weight decay: ", weight_decay)
        print("Batch size: ", batch_size)
        print("Num epochs: ", num_epochs)
        print("Noise: ", noise)
        print("Dropout: ", dropout)

        # Call the training function with the hyperparameters
        trainMain(netArch=netArch,
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
        
        accResultValid = testMain(
            netArch=netArch,
            name = "valid_250",
            checkpointName = checkpointFile
            )

        print("Validation accuracy: ", accResultValid)
        f.write("Validation accuracy: " + str(accResultValid) + "\n")

        accResultTest = testMain(
            netArch=netArch,
            name = "test_250",
            checkpointName = checkpointFile
            )
        
        print("Test accuracy: ", accResultTest)
        f.write("Test accuracy: " + str(accResultTest) + "\n")

        os.remove("checkpoint/"+checkpointFile)
    

        f.write("Noise = 0.0, Dropout = 0.0, epochs = 20, batch=6, learnrate = 0.001, weight decay = 0.0\n")
    

    # f.write("\n\n\n")
    # f.write("====================================================\n")

    
    #f.write("Optimized noise\n")
    for i in range(inputSet-1,inputSet):
        print("Training net: ", listOfNames[i])
        f.write("Training net: " + listOfNames[i] + "\n")

        netArch = listOfNets[i]
        checkpointFile = "test.pth"
        learning_rate = listOfLearningRates[i]
        weight_decay = listOfWeightDecays[i]
        batch_size = listOfBatchSizes[i]
        num_epochs = 50
        noise = 0.0
        dropout = listOfDropout[i]

        print("Hyperparameters: ")
        print("Learning rate: ", learning_rate)
        print("Weight decay: ", weight_decay)
        print("Batch size: ", batch_size)
        print("Num epochs: ", num_epochs)
        print("Noise: ", noise)
        print("Dropout: ", dropout)

        # Call the training function with the hyperparameters
        trainMain(netArch=netArch,
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
        
        accResultValid = testMain(
            netArch=netArch,
            name = "valid_250",
            checkpointName = checkpointFile
            )

        print("Validation accuracy: ", accResultValid)
        f.write("Validation accuracy: " + str(accResultValid) + "\n")

        accResultTest = testMain(
            netArch=netArch,
            name = "test_250",
            checkpointName = checkpointFile
            )
        
        print("Test accuracy: ", accResultTest)
        f.write("Test accuracy: " + str(accResultTest) + "\n")

        os.remove("checkpoint/"+checkpointFile)
    

    # f.write("\n\n\n")
    # f.write("====================================================\n")

    # f.write("Optimized droppout\n")
    for i in range(inputSet-1,inputSet):
        print("Training net: ", listOfNames[i])
        f.write("Training net: " + listOfNames[i] + "\n")

        netArch = listOfNets[i]
        checkpointFile = "test.pth"
        learning_rate = listOfLearningRates[i]
        weight_decay = listOfWeightDecays[i]
        batch_size = listOfBatchSizes[i]
        num_epochs = 50
        noise = listOfNoise[i]
        dropout = 0.0


        print("Hyperparameters: ")
        print("Learning rate: ", learning_rate)
        print("Weight decay: ", weight_decay)
        print("Batch size: ", batch_size)
        print("Num epochs: ", num_epochs)
        print("Noise: ", noise)
        print("Dropout: ", dropout)

        # Call the training function with the hyperparameters
        trainMain(netArch=netArch,
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
        
        accResultValid = testMain(
            netArch=netArch,
            name = "valid_250",
            checkpointName = checkpointFile
            )

        print("Validation accuracy: ", accResultValid)
        f.write("Validation accuracy: " + str(accResultValid) + "\n")

        accResultTest = testMain(
            netArch=netArch,
            name = "test_250",
            checkpointName = checkpointFile
            )
        
        print("Test accuracy: ", accResultTest)
        f.write("Test accuracy: " + str(accResultTest) + "\n")

        os.remove("checkpoint/"+checkpointFile)

    # f.write("\n\n\n")
    # f.write("====================================================\n")

    # f.write("Optimize epochs number\n")
    # for i in range(inputSet-1,inputSet):
    #     print("Training net: ", listOfNames[i])
    #     f.write("Training net: " + listOfNames[i] + "\n")

    #     netArch = listOfNets[i]
    #     checkpointFile = "test.pth"
    #     learning_rate = listOfLearningRates[i]
    #     weight_decay = listOfWeightDecays[i]
    #     batch_size = listOfBatchSizes[i]
    #     num_epochs = 20
    #     noise = listOfNoise[i]
    #     dropout = listOfDropout[i]

    #     print("Hyperparameters: ")
    #     print("Learning rate: ", learning_rate)
    #     print("Weight decay: ", weight_decay)
    #     print("Batch size: ", batch_size)
    #     print("Num epochs: ", num_epochs)
    #     print("Noise: ", noise)
    #     print("Dropout: ", dropout)


    #     # Call the training function with the hyperparameters
    #     trainMain(netArch=netArch,
    #         nameTrain ="train_250",
    #         nameValid = "valid_250",
    #         checkpointName = checkpointFile,
    #         learning_rate=learning_rate, 
    #         weight_decay=weight_decay, 
    #         batch_size=batch_size, 
    #         num_epochs=num_epochs, 
    #         noise=noise, 
    #         dropout=dropout,
    #         )
        
    #     accResultValid = testMain(
    #         netArch=netArch,
    #         name = "valid_250",
    #         checkpointName = checkpointFile
    #         )

    #     print("Validation accuracy: ", accResultValid)
    #     f.write("Validation accuracy: " + str(accResultValid) + "\n")

    #     accResultTest = testMain(
    #         netArch=netArch,
    #         name = "test_250",
    #         checkpointName = checkpointFile
    #         )
        
    #     print("Test accuracy: ", accResultTest)
    #     f.write("Test accuracy: " + str(accResultTest) + "\n")

    #     os.remove("checkpoint/"+checkpointFile)
    
    # f.write("\n\n\n")
    # f.write("====================================================\n")

    # f.write("Optimize batch size\n")
    # for i in range(inputSet-1,inputSet):
    #     print("Training net: ", listOfNames[i])
    #     f.write("Training net: " + listOfNames[i] + "\n")

    #     netArch = listOfNets[i]
    #     checkpointFile = "test.pth"
    #     learning_rate = listOfLearningRates[i]
    #     weight_decay = listOfWeightDecays[i]
    #     batch_size = 6
    #     num_epochs = 50
    #     noise = listOfNoise[i]
    #     dropout = listOfDropout[i]


    #     print("Hyperparameters: ")
    #     print("Learning rate: ", learning_rate)
    #     print("Weight decay: ", weight_decay)
    #     print("Batch size: ", batch_size)
    #     print("Num epochs: ", num_epochs)
    #     print("Noise: ", noise)
    #     print("Dropout: ", dropout)



    #     # Call the training function with the hyperparameters
    #     trainMain(netArch=netArch,
    #         nameTrain ="train_250",
    #         nameValid = "valid_250",
    #         checkpointName = checkpointFile,
    #         learning_rate=learning_rate, 
    #         weight_decay=weight_decay, 
    #         batch_size=batch_size, 
    #         num_epochs=num_epochs, 
    #         noise=noise, 
    #         dropout=dropout,
    #         )
        
    #     accResultValid = testMain(
    #         netArch=netArch,
    #         name = "valid_250",
    #         checkpointName = checkpointFile
    #         )

    #     print("Validation accuracy: ", accResultValid)
    #     f.write("Validation accuracy: " + str(accResultValid) + "\n")

    #     accResultTest = testMain(
    #         netArch=netArch,
    #         name = "test_250",
    #         checkpointName = checkpointFile
    #         )
        
    #     print("Test accuracy: ", accResultTest)
    #     f.write("Test accuracy: " + str(accResultTest) + "\n")

    #     os.remove("checkpoint/"+checkpointFile)
    
    # f.write("\n\n\n")
    # f.write("====================================================\n")
    

    f.write("Optimize learning rate\n")
    for i in range(inputSet-1,inputSet):
        print("Training net: ", listOfNames[i])
        f.write("Training net: " + listOfNames[i] + "\n")

        netArch = listOfNets[i]
        checkpointFile = "test.pth"
        learning_rate = 0.001
        weight_decay = listOfWeightDecays[i]
        batch_size = listOfBatchSizes[i]
        num_epochs = 50
        noise = listOfNoise[i]
        dropout = listOfDropout[i]


        print("Hyperparameters: ")
        print("Learning rate: ", learning_rate)
        print("Weight decay: ", weight_decay)
        print("Batch size: ", batch_size)
        print("Num epochs: ", num_epochs)
        print("Noise: ", noise)
        print("Dropout: ", dropout)



        # Call the training function with the hyperparameters
        trainMain(netArch=netArch,
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
        
        accResultValid = testMain(
            netArch=netArch,
            name = "valid_250",
            checkpointName = checkpointFile
            )

        print("Validation accuracy: ", accResultValid)
        f.write("Validation accuracy: " + str(accResultValid) + "\n")

        accResultTest = testMain(
            netArch=netArch,
            name = "test_250",
            checkpointName = checkpointFile
            )
        
        print("Test accuracy: ", accResultTest)
        f.write("Test accuracy: " + str(accResultTest) + "\n")

        os.remove("checkpoint/"+checkpointFile)
    
    f.write("\n\n\n")
    f.write("====================================================\n")
    f.write("END\n")


