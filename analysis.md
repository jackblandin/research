## QLearner
05/07/2019
notebook: `notebooks/tiger_env/qlearner.ipynb`

The final Q values are:

```
24999           OPEN LEFT | OPEN RIGHT | LISTEN
                --------- | ---------- | ------
GROWL LEFT         -13.78 |      -6.69 | -11.04
GROWL RIGHT:        -5.98 |     -14.27 | -11.26
START:             -12.47 |     -12.41 |  -5.85
END:                 0.72 |       0.84 |   0.17
```

We can see that the final policy is to

observation | action
--- | ---
START | LISTEN
GROWL LEFT | OPEN RIGHT
GROWL RIGHT | OPEN LEFT

Assuming epsilon is zero, the expected value of taking action OPEN LEFT or OPEN RIGHT prior to receiving any observation is

```
.5(-100) + .5(10) = -45
```
The time horizon is 10 steps (set by the `max_steps` parameter in the `utils.play_one` method). Therefore, the expected value for taking the LISTEN action prior to receiving any observation is

```
-1(10) = -10
```

Therefore, it makes sense that the optimal actions for GROWL LEFT and GROWL RIGHT are LISTEN, since -10 is greater than -45. If we changed the listen reward to be -50, then we'd expect OPEN LEFT or OPEN RIGHT to become the best action prior to receiving any observations. In fact, it does:

```
24999           OPEN LEFT | OPEN RIGHT | LISTEN
                --------- | ---------- | ------
GROWL LEFT           -4.2 |      -4.07 |  -4.21
GROWL RIGHT:        -4.49 |      -4.74 |  -4.67
START:             -45.32 |     -46.93 |  -47.2
END:                -0.09 |       0.03 |  -0.62
```


When we receive an observation, it must be either GROWL LEFT or GROWL RIGHT, and if we take action OPEN RIGHT or OPEN LEFT respectively, then the expected value is

```
.85(10) + -100(.15) = -6.5
```

Therefore, sincye -6.5 > -10, it makes sense that the optimal action is to OPEN LEFT or OPEN RIGHT, rather than LISTEN. If we decrease the `max_steps` parameter to be less than 6, however, we should expect the optimal action to always be to LISTEN. In fact, if we set `max_steps` to 5, we see the following Q values

```
24999           OPEN LEFT | OPEN RIGHT | LISTEN
                --------- | ---------- | ------
GROWL LEFT         -10.36 |      -6.02 |  -8.98
GROWL RIGHT:        -7.03 |      -10.6 |  -9.08
START:             -10.89 |     -10.32 |  -7.48
END:                 0.32 |      -0.96 |   0.54
```

**OPEN QUESION** - Whey don't we see above hypothesis happening?


## QLearnerSeq
05/07/2019
notebook: `notebooks/tiger_env/qlearner-seq.ipynb`

As expected, as we increase the number of observation sequences in memory, the performance improves:

sequence length | avg reward after policy convergence
--- | ---
1 | -8.1512
2 | 2.3695
3 | 3.5433
4 | 2.8276
5 | 2.5019
6 | -2.4601

**OPEN QUESTION** - why does performance start to decrease at seq length = 4?
