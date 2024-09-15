import numpy as np
import random
import pickle
import matplotlib.pyplot as plt


class Neurona:

    def __init__(self, structure, charge_data = 0, type = 1):
        self.z = 0
        self.a = 0
        self.pesos = np.array(charge_data[0]) if charge_data else np.random.rand(structure)
        self.sesgo = charge_data[1] if charge_data else random.random()
        self.inputs = 0


    def relu(self, z):
        return z if z > 0 else 0
    
    def relu_deri(self, z):
        return 1 if z > 0 else 0
    
    def sigm(self, z):
        return 1 / (1 + np.e ** - z)
    
    def sigm_deri(self, z):
        return self.sigm(z) * (1 - self.sigm(z))
    
    def give_value(self, inputs):
        self.inputs = inputs
        self.z = self.inputs @ self.pesos + self.sesgo 
        self.a = self.relu(self.z)
        return self.a

class Red:

    def __init__(self, structure: list, confName: str = "data", ultCapa = 0) -> None:
        self.output = 0
        self.red = []
        self.act_ant = []
        self.mse = lambda pred, solu: (pred - solu)**2/2
        self.mse_deri =  lambda pred, solu: pred - solu
        self.nose = [lambda movs, win: movs**2 if win else 0, lambda movs, win: 2*movs if win else 0]

        self.correcto = []
        self.red_sigmas = []
        self.fileName = confName

        try:
            with open(self.fileName, "rb") as data:
                data = pickle.load(data)
                for l, dato in enumerate(data):
                    if len(dato) != structure[l+1]:
                        raise ValueError
                print("CON")
                self.red = data
        except:           
            print("JLKJÑLK")
            for l, columns in enumerate(structure[1:]):
                self.red.append([])
                for rows in range(columns):
                    self.red[l].append(Neurona(structure[l], type=1))

    

    def propagar(self, inputs):
        self.act_ant.clear()
        for l, columns in enumerate(self.red):
            self.act_ant.append(np.array([]))

            if l == 0:
                for neu in columns:
                    self.act_ant[l] = np.append(self.act_ant[l], neu.give_value(inputs))

            else:
                for neu in columns:
                    self.act_ant[l] = np.append(self.act_ant[l], neu.give_value(self.act_ant[l-1]))
        

    def entrenar(self, inputs: np, solus, ciclos: int = 1000, ln: float = 0.05, parametrosDC = None, dcoste = 0):
        
        inverse_red = self.red[::-1]
        print(inverse_red)
        for _ in range(ciclos):
            for parametro in range(len(inputs)):
                self.propagar(inputs[parametro])
                for l, columns in enumerate(inverse_red):
                    sigmas = np.zeros(len(inverse_red[l][0].pesos))
                    
                    if l == 0:
                        self.red_sigmas = np.zeros(len(inverse_red[l][0].pesos))

                        for index, neu in enumerate(columns):

                            sigma = self.mse_deri(neu.a, solus[parametro]) * neu.relu_deri(neu.z)
                            neu.pesos = neu.pesos - (sigma * neu.inputs) * ln
                            neu.sesgo = neu.sesgo - sigma * ln

                            
                            self.red_sigmas += (sigma[0] * neu.pesos)



                        
                    else:
                        for index, neu in enumerate(columns):

                            sigma = self.red_sigmas[index] * neu.relu_deri(neu.z)
                            neu.pesos = neu.pesos - (sigma * neu.inputs) * ln
                            neu.sesgo = neu.sesgo - sigma * ln
                            sigmas += (sigma * neu.pesos)

                    self.red_sigmas = sigmas.copy()


        self.save_datos()

        

    def save_datos(self):
        with open(self.fileName, "wb") as conf:
            data = self.red
            pickle.dump(data, conf)
            



def lineReg():
    x = []
    y = []
    for i in range(100):
        num = random.random()*50 + i
        x.append([i])
        y.append([num])

    plt.scatter(x, y)
    return np.array(x), np.array(y)

if __name__ == "__main__":
    
    mi_red = Red(structure = [1, 2, 2, 1], confName="data")
    #mi_red.entrenar(inputs=np.array([[20],[40], [80]]), solus=np.array([[40],[80],[160]]), ciclos=1000, ln=0.001)

    
    mi_red.propagar([20])

    print(mi_red.red[-1][0].a, "jjjjj")
    mi_red.propagar([40])
    print(mi_red.red[-1][0].a, "jjjjj")
    
    mi_red.propagar([80])
    print(mi_red.red[-1][0].a, "jjjjj")



