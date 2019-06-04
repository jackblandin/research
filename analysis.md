# QLearner
* 05/07/2019
* notebook: notebooks/tiger_env/qlearner.ipynb

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

### Why don't Q values converge to actual expected values/How do alpha and gamma effect converged Q values?

One interesting point of observation is to notice that the final Q values are dependent on how `alpha` and `gamma` are set. To understand how these values influnce the final Q values, we can look at the TD(0) update and then an example. For simplicity's sake, assume that we have a fixed policy and are updating the Q values according to the policy.

```
Q[s0,a] = Q[s0,a0]  + a*( g*(r  + Q[s1,a1])     - Q[s0,a0] )
        = Q[s0,a0]  + a*(g*r    + g*Q[s1,a1]    - Q[s0,a0] )
        = Q[s0,a0]  + a*g*r     + a*g*Q[s1,a1]  - a*Q[s0,a0]
```

Suppose we are starting with all zero Q values.

```
                OPEN LEFT | OPEN RIGHT | LISTEN
                --------- | ---------- | ------
GROWL LEFT              0 |          0 |      0
GROWL RIGHT:            0 |          0 |      0
START:                  0 |          0 |      0
END:                    0 |          0 |      0
```

Our fixed policy says to choose LISTEN for the START observation, so we choose LISTEN as our initial action and receive observation GL. Then our first belief update is

```
Q[St, L] = Q[St,L]  + a*g*-1    + a*g*Q[GL,OL])     - a*Q[St,L] )
          = 0       + -1*a*g    + a*g*0             - a*0
          = -1*a*g
```

Note: Our fixed policy says to take action OL when receiving observation GL.

Then, our updated Q values are

```
                OPEN LEFT | OPEN RIGHT | LISTEN
                --------- | ---------- | ------
GROWL LEFT              0 |          0 |      0
GROWL RIGHT:            0 |          0 |      0
START:                  0 |          0 | -1*a*g
END:                    0 |          0 |      0

Q[GL, OL] = Q[GL,OL]  + a*g*-100    + a*g*Q[E,*])   - a*Q[GL,OL] )  # action irrelevant
          = 0         + -100*a*g      + a*g*0         - a*0
          = -100*a*g


                OPEN LEFT | OPEN RIGHT | LISTEN
                --------- | ---------- | ------
GROWL LEFT       -100*a*g |          0 |      0
GROWL RIGHT:            0 |          0 |      0
START:                  0 |          0 | -1*a*g
END:                    0 |          0 |      0
```

Now we restart a new episode and again choose to LISTEN

```
Q[St, L]  = Q[St,L] + a*g*-1    + a*g*Q[GL,OL])     - a*Q[St,L]
          = -1*a*g  + -1*a*g    + a*g*(-100*a*g)    - a*(-1*a*g)
          = -1*a*g  + -1*a*g    + -100*a^2*g^2      - -1*a^2*g
          = -2*a*g              - 100*a^2*g^2       + a^2*g


                OPEN LEFT | OPEN RIGHT | LISTEN
                --------- | ---------- | ------
GROWL LEFT       -100*a*g |          0 |      0
GROWL RIGHT:            0 |          0 |      0
START:                  0 |          0 | -2*a*g - 100*a^2*g^2 + a^2*g
END:                    0 |          0 |      0
```

and again we receive TL and take action OL (environment is conveniently redundant)

```

Q[GL, OL] = Q[GL,OL]  + a*g*-100    + a*g*Q[E,*])   - a*Q[GL,OL] )  # action irrelevant
          = -100*a*g  + -100*a*g      + a*g*0       - -100*a^2*g
          = -200*a*g + 100*a^2*g


                           OPEN LEFT | OPEN RIGHT | LISTEN
                           --------- | ---------- | ------
GROWL LEFT      -200*a*g + 100*a^2*g |          0 |      0
GROWL RIGHT:                       0 |          0 |      0
START:                             0 |          0 | -2*a*g - 100*a^2*g^2 + a^2*g
END:                               0 |          0 |      0
```

and we repeat the whole episode process once more

```
Q[St, L]  = Q[St,L] + a*g*-1    + a*g*Q[GL,OL])     - a*Q[St,L]
          = -2*a*g - 100*a^2*g^2 + a^2*g  + -1*a*g    + a*g*(-200*a*g)    - -2*a^2*g - 100*a^3*g^2 + a^3*g
          = -2*a*g - 100*a^2*g^2 + a^2*g  + -1*a*g    + -200*a^2*g^2      +2*a^2*g - 100*a^3*g^2 + a^3*g

                           OPEN LEFT | OPEN RIGHT | LISTEN
                           --------- | ---------- | ------
GROWL LEFT      -200*a*g + 100*a^2*g |          0 |      0
GROWL RIGHT:                       0 |          0 |      0
START:                             0 |          0 | -2*a*g - 100*a^2*g^2 + a^2*g  + -1*a*g    + -200*a^2*g^2      +2*a^2*g - 100*a^3*g^2 + a^3*g
END:                               0 |          0 |      0


Q[GL, OL] = Q[GL,OL]  + a*g*-100    + a*g*Q[E,*])   - a*Q[GL,OL] )  # action irrelevant
          = -200*a*g + 100*a^2*g  + -100*a*g      + a*g*0       +200*a^2*g + 100*a^3*g
          = -300*a*g + 100*a^2*g  +200*a^2*g + 100*a^3*g


                                                              OPEN LEFT | OPEN RIGHT | LISTEN
                                                              --------- | ---------- | ------
GROWL LEFT       -200*a*g + 100*a^2*g  - 100*a*g +200*a^2*g + 100*a^3*g |          0 |      0
GROWL RIGHT:                                                          0 |          0 |      0
START:                                                                0 |          0 | -2*a*g - 100*a^2*g^2 + a^2*g
END:                                                                  0 |          0 |      0
```

Now let's plugin values for `a` and `g` to see how this looks. Let `a = .1`, and `g = .9`.

```
                                                              OPEN LEFT | OPEN RIGHT | LISTEN
                                                              --------- | ---------- | ------
GROWL LEFT       -18+ 100*a^2*g  - 100*a*g +200*a^2*g + 100*a^3*g |          0 |      0
GROWL LEFT: -200 + .9 - 9 + 1.8 +.09 =
GROWL RIGHT:                                                          0 |          0 |      0
START:                                                                0 |          0 | -2*a*g - 100*a^2*g^2 + a^2*g
END:                                                                  0 |          0 |      0
```

For the OL update, since next state is always 0, the update is simply
```
Q[s0,a] = Q[s0,a0]  + a*( g*(r  + Q[s1,a1])     - Q[s0,a0] )
        = Q[s0,a0] + a*g*r - a*Q[s0,a0]
```
So it does not depned on `g` at all. But if `a` is larger, and we assume that Q[GL,OL) is monotonically increasing in magnitude (which it is due to the fixed policy), then we see that the update (the right two terms) become... I think it will oscilate between smaller and smaller positive and negative value??? TODO: TEST IT OUT USING A FIXED POLICY
If each update decreases in magnitude, then a larger alpha will allow for larger initial udpates, which should result in larger magnitudes of Q values. One might argue that if alpha is larger, then the subsequent negative updates will also be larger. This is true, but I think that the larger alpha will still leave a bigger gap.

# QLearnerSeq
* 05/07/2019
* notebook: notebooks/tiger_env/qlearner-seq.ipynb

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

