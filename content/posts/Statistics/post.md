---
date: '2026-03-05T17:10:26+09:00'
draft: false
title: 'Statistical Notes: Questions, Clarifications, and Intuition'

tags: ["Statistics"]
---
# Introduction
This blog reflects my journey through Machine Learning, Reinforcement Learning, and Robotics. While my current work spans these areas, my academic roots are in Statistics, a foundation that continues to shape how I think about problems. From a theoretical standpoint, I believe Statistics remains the core foundation of modern machine learning.

Over time, while studying statistics, I often encountered concepts that felt confusing, subtle, or easy to misunderstand. Some of them became clearer after revisiting the ideas multiple times or thinking about them from different perspectives, while others are still things I occasionally find myself reflecting on.

In this series of notes, I plan to write down those questions, confusions, and small clarifications, both as a personal record and as a way to organize my own understanding of statistical concepts.

# Uncategorized (yet)

## What exactly is the random variable?
First, I want to clarify this since notations related to it are sometimes confusing. **Random variable is a function from a sample space to the real numbers. It is a function that maps each outcome of a random experiment to a real number**.
$$ X : \Omega \rightarrow \mathbb{R} $$ 

Although it is called a variable, it is important to remember that a random variable is actually a function, not a variable in the usual sense.

For example, if we flip two coins, the sample space is {HH, HT, TH, TT}. And the random variable X is the number of heads. $$ X(HH) = 2, X(HT) = 1, X(TH) = 1, X(TT) = 0 $$
$$ X : \\{HH, HT, TH, TT\\} \rightarrow \\{0, 1, 2\\} $$
So, random variable is a function, not a variable.

$$ P(X=x) $$
At first glance, this can be confusing. What exactly is $x$? Is $x$ an input of the function $X$? No. The input of $X$ is an outcome of the random experiment. So, $x$ is the output of the function $X$. More precisely,
$$ P(X=x) = P(\\{ \omega \in \Omega | X(\omega) = x \\}) $$
For example, $P(X=1) = P(\\{ \omega \in \Omega | X(\omega) = 1 \\}) = P(\\{ HT, TH \\}) = 1/2$.

## Intuitively, Why Does Integrating $-F(x)$ and $1-F(x)$ Give $E(X)$?

Typically, the definition of expected value is given as:
$$ E(X) = \int_{-\infty}^{\infty} x f(x) dx \tag{1} $$ 
But there's another way to get $E(X)$ using the cumulative distribution function $F(x)$.
$$ E(X) = -\int_{-\infty}^{0} F(x) dx + \int_{0}^{\infty} (1-F(x)) dx \tag{2} $$
Intuitively, (1) is quite straightforward. It is the weighted average of all possible values of $X$ by their probabilities. If you think it in visual way, it is like this:
