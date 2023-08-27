import perceval as pcvl
import sympy as sp
import numpy as np
from perceval.components import BS, PS
import random
import matplotlib.pyplot as plt


bs_rx = BS.Rx()

results = []
noise_scaling = []

source = pcvl.Source(emission_probability=0.80, multiphoton_component=0.1, losses=0.4)

source = pcvl.Source()
states = {
    pcvl.BasicState([1, 0]): "0",
    pcvl.BasicState([0, 1]): "1",
}

number = 0
for x in range(1, 10, 2):
    print(x)
    number+=1
    noise_scaling.append(number)

    test1 = pcvl.Circuit(2, name="test1")

    for i in range(x):
        noise_angle_x_1 = random.uniform(0.001, np.pi/20)
        noise_angle_x_2 = random.uniform(0.001, np.pi/20)
        noise_angle_x_3 = random.uniform(0.001, np.pi/20)
        noise_angle_x_4 = random.uniform(0.001, np.pi/20)
        noise_angle_x_5 = random.uniform(0.001, np.pi/20)
        noise_angle_x_6 = random.uniform(0.001, np.pi/20)

        test1.add((0, 1), BS.Rx(theta= np.pi - noise_angle_x_1 ))

        #test1.add((0, 1), BS.Rx(theta= np.pi - noise_angle_x_3 ))
        #test1.add((0, 1), BS.Rx(theta= np.pi - noise_angle_x_4 ))
        #test1.add((0, 1), BS.Rx(theta= np.pi - noise_angle_x_5 ))
        #test1.add((0, 1), BS.Rx(theta= np.pi - noise_angle_x_6 ))
        test1.add((0, 1), BS.Ry(theta=np.pi - noise_angle_x_3))

        test1.add((0, 1), BS.Rx(theta=np.pi - noise_angle_x_2))

        test1.add((0, 1), BS.Rx(theta=np.pi - noise_angle_x_5))
        test1.add((0, 1), BS.Rx(theta=np.pi - noise_angle_x_6))

        test1.add((0, 1), BS.Ry(theta=np.pi - noise_angle_x_4))

        #bs_rx_error = BS.Rx(theta=(np.pi) - noise_angle_x_4)




    #pcvl.pdisplay(test1, recursive = True)
    p = pcvl.Processor("SLOS", test1, source)
    #backend = pcvl.BackendFactory.get_backend("Naive")


    #ca = pcvl.algorithm.Analyzer(p, states)
    #ca.compute(expected={"0": "0", "1": "1"})
    #pcvl.pdisplay(ca)
    #backend.set_circuit(test1)
    #backend.set_input_state(pcvl.BasicState([1, 0]))
    #print("prob")
    #print(backend.probability(pcvl.BasicState([1,0])))
    # Gives the source distribution

    p.with_input(pcvl.BasicState([1, 0]))

    # pcvl.pdisplay(QPU.source_distribution, precision=1e-4)

    # Gives the output distribution
    #p.set_postselection(pcvl.PostSelect("[0] == 1 & [1] ==0"))
    #p.probs().keys()
    #p.probability(pcvl.BasicState([1,0]))
    output_distribution = p.probs()["results"]
    pcvl.pdisplay(output_distribution, max_v=10)
    #print(output_distribution.keys())

    for output_bs, prob in output_distribution.items():
        output_state = output_bs
        if (output_state[0] == 1 and output_state[1] == 0):
            results.append(prob)
            print("testing")
            print(prob)
            #print(list(output_distribution.values())[0])

#print(results)
out = []
for i in range(len(noise_scaling)):
    u = [noise_scaling[i],results[i]]
    out.append(u)
print(out)

p_2 = np.polyfit(noise_scaling,results,deg=2)
p_3 = np.polyfit(noise_scaling,results,deg=3)
p_4 = np.polyfit(noise_scaling,results,deg=1)

x_new = np.array([0])
y_2 = np.polyval(p_2,x_new)
y_3 = np.polyval(p_3,x_new)
y_4 = np.polyval(p_4,x_new)
print(y_2, y_3, y_4)

#results.insert(0,y_new)
#noise_scaling.insert(0,0)

#plt.xticks(range(min(noise_scaling), max(noise_scaling)+1))
plt.plot(noise_scaling,results, "o--", linewidth=2.5, markersize=7)
plt.plot(0,y_2, "o", markersize=7, label = "poly 2")
plt.plot(0,y_3, "o", markersize=7, label = "poly 3")
plt.plot(0,y_4, "o", markersize=7, label = "poly 1")
plt.xticks(size=12)
plt.yticks(size=12)
plt.xlabel('Noise Scaling', size= 12)
plt.ylabel('Expectation Value', size=12)
plt.legend()
plt.grid()
plt.show()