from configuration import ALG

if ALG == "APE_X":
    from APE_X.Learner import Learner

elif ALG == "R2D2":
    from R2D2.Learner import Learner

elif ALG == "APE_X_CPT":
    from APE_X_Crypto.Learner import Learner

elif ALG == "R2D2_CPT":
    from R2D2_Crypto.Learner import Learner

else:
    raise RuntimeError("!!")

if __name__ == "__main__":

    l = Learner()
    l.run()