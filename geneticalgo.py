
import random
print("cheeks")

if __name__ == "__main__":
    population = 100

    def encode():
        
        char1 = random.choice([-1,1])
        char2 = random.randint(0,9)
        char3 = random.randint(0,9)
        char4 = random.randint(0,9)
        char5 = random.randint(0,9)
        return [char1,char2,char3,char4,char5]
        
    individual = (encode())
    
    print("".join(str(x) for x in individual))


