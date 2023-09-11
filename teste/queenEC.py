#EXEMPLO UTILIZANDO O N-Rainhas PERMUTADO
#
#EXEMPLO ARQUIVO basicECParam
#N=8
#POP=10
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ paralelizar com profiler
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os
from functools import partial
import tqdm
from multiprocessing.pool import ThreadPool

def getInitialPopulation(n,pop):
    population = []
    for _ in range(pop):
        chromo = list(range(n))
        random.shuffle(chromo)
        population.append(chromo)

    return population

def fitnessLoop(n,population,p):
    worstFitness = n*(n-1)//2
    collision = 0
    for i in range(n):
        for j in range(i-1,-1,-1):
            if((population[p][j] == population[p][i] + (i-j)) or (population[p][j] == population[p][i] - (i-j))):
                collision += 1
        for j in range(i+1,n,1):
            if((population[p][j] == population[p][i] + (j-i)) or (population[p][j] == population[p][i] - (j-i))):
                collision += 1
    return worstFitness-(collision//2)
                    
def populationFitness(n,pop,population):
    populationFitness2 = []
    with Pool() as pool:
        print(pop)
        populationFitness2 = pool.map(partial(fitnessLoop,n,population),range(pop))
        print("FINISH",populationFitness2)
    return populationFitness2

def bestFitnessObj(n,pop,population):
    popFitness = populationFitness(n,pop,population)
    avgChromoEval = sum(popFitness) / len(popFitness)
    bestChromoConfig = population[popFitness.index(max(popFitness))]
    bestChromoEval = max(popFitness)
    return {
        "bestConfig": bestChromoConfig,
        "bestEval": bestChromoEval,
        "avgEval": avgChromoEval
    }

def setEliteSolution(n,pop,population):
    popFitness = populationFitness(n,pop,population)
    maxIndex = popFitness.index(max(popFitness))
    return population[maxIndex].copy()

def insertElite(n,pop,population,elite):
    popFitness = populationFitness(n,pop,population)
    minIndex = popFitness.index(min(popFitness))
    population[minIndex] = elite.copy()
    return population

def rouletteSelection(n,pop,population):
    popFitness = populationFitness(n,pop,population)
    popFitnessSum = sum(popFitness)
    rouletteDivision = []
    for i in range(len(population)):
        rouletteDivision.append(popFitness[i]/popFitnessSum)
    roulettePairs = []
    #Loop de geração dos pares da população intermediária
    for _ in range(len(population)//2):
        limit = random.SystemRandom().uniform(0, 1)
        limitSum = 0
        #retira o primeiro componente do par e prepara as listas para a seleção do próximo (sem reposição)
        for i in range(len(rouletteDivision)):
            if(limit <= rouletteDivision[i] + limitSum):
                roulettePairs.append(population[i])
                popAux = population.copy()
                popAux.pop(i)
                popFitnessAux = popFitness.copy()
                popFitnessAux.pop(i)
                popFitnessSumAux = sum(popFitnessAux)
                rouletteDivisionAux = []
                for j in range(len(popAux)):
                    rouletteDivisionAux.append(popFitnessAux[j]/popFitnessSumAux)
                break
            else:
                limitSum += rouletteDivision[i]
        limit = random.SystemRandom().uniform(0, 1)
        limitSum = 0
        for j in range(len(rouletteDivisionAux)):
            if(limit <= rouletteDivisionAux[j] + limitSum):
                roulettePairs.append(population[j])
                break
            else:
                limitSum += rouletteDivisionAux[j]
    return roulettePairs

def tournamentSelection(n,pop,population,k,kp):
    popFitness = populationFitness(n,pop,population)
    roulettePairs = []
    for _ in range(len(population)):
        selectedChromo = [x for x in range(len(population))]
        bestChromoIndex = random.SystemRandom().randint(0, (len(selectedChromo)-1))
        bestChromo = selectedChromo[bestChromoIndex]
        selectedChromo.pop(bestChromoIndex)
        worstChromo = bestChromo
        for _ in range(k-1):
            chromoIndex = random.SystemRandom().randint(0, (len(selectedChromo)-1))
            chromo = selectedChromo[chromoIndex]
            selectedChromo.pop(chromoIndex)
            if(popFitness[chromo] > popFitness[bestChromo]):
                bestChromo = chromo
            if(popFitness[chromo] < popFitness[worstChromo]):
                worstChromo = chromo
        limitBest = random.SystemRandom().uniform(0, 1)
        if(kp >= limitBest):
            roulettePairs.append(population[bestChromo])
        else:
            roulettePairs.append(population[worstChromo])
    return roulettePairs

def crossoverPMX(population,p):
    for i in range(0,len(population),2):
        #porcentagem de ocorrência
        crossoverCheck = random.SystemRandom().uniform(0, 1)
        if(crossoverCheck<=p):
            #janela de corte
            firstCutAux = random.SystemRandom().randint(0,(len(population[i])))
            secondCutAux = random.SystemRandom().randint(0,(len(population[i])))
            firstCut = min(firstCutAux,secondCutAux)
            secondCut = max(firstCutAux,secondCutAux)
            #inversão de janela nas listas
            firstListCut = population[i][firstCut:secondCut]
            secondListCut = population[i+1][firstCut:secondCut]
            population[i][firstCut:secondCut] = secondListCut
            population[i+1][firstCut:secondCut] = firstListCut

            #criação do dicionário das relações
            firstDictionarySwap = {}
            secondDictionarySwap = {}
            for j in range(len(secondListCut)):
                firstDictionarySwap[secondListCut[j]] = firstListCut[j]
            for j in range(len(firstListCut)):
                secondDictionarySwap[firstListCut[j]] = secondListCut[j]

            #correção permutação
            for j in range(0,firstCut):
                for k in range(len(secondListCut)):
                    if(population[i][j] == secondListCut[k]):
                        population[i][j] = firstListCut[k]
                        while(population[i][j] in secondListCut):
                            population[i][j] = firstDictionarySwap[population[i][j]]
                    if(population[i+1][j] == firstListCut[k]):
                        population[i+1][j] = secondListCut[k]
                        while(population[i+1][j] in firstListCut):
                            population[i+1][j] = secondDictionarySwap[population[i+1][j]]
            for j in range(secondCut,len(population[i])):
                for k in range(len(secondListCut)):
                    if(population[i][j] == secondListCut[k]):
                        population[i][j] = firstListCut[k]
                        while(population[i][j] in secondListCut):
                            population[i][j] = firstDictionarySwap[population[i][j]]
                    if(population[i+1][j] == firstListCut[k]):
                        population[i+1][j] = secondListCut[k]
                        while(population[i+1][j] in firstListCut):
                            population[i+1][j] = secondDictionarySwap[population[i+1][j]]
    return population

def mutation(population,p):
    for i in range(len(population)):
        for j in range(len(population[i])):
            mutationCheck = random.SystemRandom().uniform(0, 1)
            if(mutationCheck<=p):
                swapPosition = random.SystemRandom().randint(0,(len(population[i])-1))
                while(swapPosition == j):
                    swapPosition = random.SystemRandom().randint(0,(len(population[i])-1))
                geneAux = population[i][j]
                population[i][j] = population[i][swapPosition]
                population[i][swapPosition] = geneAux
    return population

if __name__ == "__main__":
    file = open("basicECParamEX2.txt", "r")
    fileData = file.readlines()
    N = int(fileData[0].replace("\n","").split("=")[1])
    POP = int(fileData[1].replace("\n","").split("=")[1])

    population = getInitialPopulation(N,POP)
    ger = 50 
    bestSolutionList = []
    bestSolutionConfigList = []
    avgSolutionList = []
    iterationPlot = []
    testFitness = bestFitnessObj(N,POP,population)
    bestSolutionList.append(testFitness["bestEval"])
    bestSolutionConfigList.append(testFitness["bestConfig"])
    avgSolutionList.append(testFitness["avgEval"])
    for i in range(ger):
        eliteSolution = setEliteSolution(N,POP,population)
        population = rouletteSelection(N,POP,population)
        population = crossoverPMX(population,0.9)
        population = mutation(population,0.05)
        population = insertElite(N,POP,population,eliteSolution)
        testFitness = bestFitnessObj(N,POP,population)
        bestSolutionList.append(testFitness["bestEval"])
        bestSolutionConfigList.append(testFitness["bestConfig"])
        avgSolutionList.append(testFitness["avgEval"])
        iterationPlot.append(i)

    print(bestSolutionList,bestSolutionConfigList,avgSolutionList)

    fig, ax1 = plt.subplots()
    ax1.plot(bestSolutionList, color='blue', label='Melhor solução')
    ax1.set_xlabel('Iterações')
    ax1.set_ylabel('Melhor solução', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a twin axis for the High_minus_Low variable
    ax2 = ax1.twinx()
    ax2.plot(avgSolutionList, color='green', label='Média das soluções')
    ax2.set_ylabel('Média das soluções', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # Add a title and show the plot
    plt.title('Gráfico de execução do problema dos rádios')
    plt.show()