import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

#La capa de entrada no se representa, pero está


class Neurona:

    def __init__(self, structure, charge_data = 0, type = 1):
        self.z = 0
        self.a = 0
        self.pesos = np.array(charge_data[0]) if charge_data else np.random.rand(structure) 
        self.sesgo = charge_data[1] if charge_data else random.random()
        #print(self.sesgo, structure)
        self.inputs = 0
        self.actType = type

    def act(self):
        if self.actType:
            return self.z if self.z > 0 else 0
        else:
            return 1 / (1 + np.e ** -self.z)

    def deriAct(self):
        if self.actType:
            return 1 if self.z > 0 else 0
        else:
            return (np.e** -self.z)/(1+ np.e** -self.z)**2
        
    def give_value(self, inputs):
        #print(inputs, self.pesos)
        self.inputs = inputs
        self.z = self.inputs @ self.pesos + self.sesgo 
        self.a = self.act()
        #print(self.pesos, self.sesgo, self.a)
        return self.a

class Red:

    def __init__(self, structure: list, confName: str = "data", ultCapa = 0) -> None:
        self.output = 0
        self.red = []
        self.act_ant = []
        self.coste = [lambda pred, solu: (solu - pred)**2, lambda movs, win: movs**2 if win else 0]
        self.deriCoste = [lambda pred, solu: pred - solu, lambda movs, win: 2*movs if win else 0]

        self.correcto = lambda a: a+155.15
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
        #///////
        for _ in range(ciclos):
            for parametro in range(len(inputs)):
                self.propagar(inputs[parametro])
                for l, columns in enumerate(self.red[::-1]):
                    sigmas = []
                    if l == 0:
                        for neu in columns:
                            #sigma = self.deriCoste(neu.a, self.correcto(inputs)) * neu.deriAct() * ln
                            #print(self.deriCoste,dcoste, parametrosDC, 89789070897)
                            sigma = self.deriCoste[dcoste](neu.a , solus[parametro]) * neu.deriAct()
                            neu.pesos = neu.pesos - (sigma * neu.inputs) * ln
                            #print(self.deriCoste(neu.a, self.correcto(inputs)), neu.deriAct())
                            neu.sesgo = neu.sesgo - sigma * ln
                            #print(sigma)
                            sigmas.append(sigma)
                            #print(neu.pesos, neu.sesgo, neu.a)
                    else:
                            sigma = sum(self.red_sigmas) * neu.deriAct() 
                            neu.pesos = neu.pesos - (sigma * neu.inputs) * ln
                            neu.sesgo = neu.sesgo - sigma * ln
                            sigmas.append(sigma)

                    self.red_sigmas.append(np.mean(sigmas))
                self.red_sigmas.clear()

        #self.save_datos()
        #print(self.red[0][0].a, self.act_ant)
        #print("OUTPUT" , self.red[-1][0].a, self.correcto(inputs))
        

    def save_datos(self):
        data = []
        for l, c in enumerate(self.red):
            data.append([])
            for f in c:
                #print(f.sesgo)
                data[l].append([f.pesos.tolist(), f.sesgo])
        #print(data)

        with open(self.fileName, "wb") as conf:
            pickle.dump(self.red, conf)
            #conf.write(json.dumps(data))


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
    """
    estructura_red = [2,2]
    user_input = np.array([27]) #????
    mi_red = Red(structure = [1, 4, 3], confName="data")
    #print(mi_red.red[-1][0].sesgo)
    mi_red.entrenar(np.array([1]), 100, 0.05, [[[1], 152]])
    mi_red.propagar(np.array([1]))
    print(mi_red.red[-1][0].a)
   """
    #x, y = lineReg()
    mi_red = Red(structure = [1, 4, 4, 2], confName="data")
    mi_red.entrenar(inputs=np.array([[20],[40], [80]]), solus=np.array([[40],[60],[20]]), ciclos=100, ln=0.00001)
    #print(mi_red.red[-1][0].pesos)
    #print(mi_red.red[-1][0].sesgo)
    mi_red.propagar([1])

    print(mi_red.red[-1][0].a, "jjjjj")
    mi_red.propagar([2])

    print(mi_red.red[-1][0].a)
    mi_red.propagar([3])

    print(mi_red.red[-1][0].a)
"""    mi_red.propagar(np.array([0]))
    p1= mi_red.red[-1][0].a
    mi_red.propagar(np.array([100]))
    p2= mi_red.red[-1][0].a
    print(p1, p2)
    plt.show()
"""
    #mi_red.save_datos()
    #mi_red.save_datos()


