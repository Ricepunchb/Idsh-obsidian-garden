---
title: Combinatorics
alias: 조합론
publish: true
date: 2026-09-22
tags:
  - Math
  - Probability
---

# 1. 조합과 순열

>[!attention] Combination (조합)
>$n$개 중 $k$개를 순서 상관없이 뽑는 경우의 수
>$$\binom{n}{k}=\dfrac{n!}{k!(n-k)!}=\dfrac{P(n,k)}{k!}$$

$k$개짜리 집합 하나는 $k!$가지 순서로 나열할 수 있으므로, 순열의 수를 $k!$로 나누면 순서를 무시한 조합의 수가 된다는 관계가 성립한다.

>[!attention] Permutation (순열)
>$n$개 중 $k$개를 순서를 구분해서 뽑아 나열하는 경우의 수
>$$P(n,k)=\dfrac{n!}{(n-k)!}$$

# 2. 다항계수 (Multinomial coefficient)

$n$개의 물건을 $1$종류는 $n_1$개, $2$종류는 $n_2$개, ... , $m$종류는 $n_m$개로 나눠 배열하는 경우의 수 ($n_1+\cdots+n_m=n$):
$$\binom{n}{n_1,\dots,n_m}=\dfrac{n!}{n_1!\cdots n_m!}$$

전체 $n$개를 일단 다 구분해서 나열($n!$)한 다음, 같은 종류 안에서의 순서는 구분할 필요가 없으니 각 종류별로 $n_i!$만큼 중복 카운트된 것을 나눠서 제거한다고 생각하면 된다.

# 3. Derangement (완전순열)

**derangement**란 $n$개를 나열했을 때 **어느 자리도 제자리가 아닌** 순열을 말한다 (고정점이 하나도 없는 순열). 예를 들어 $n$명이 선물을 하나씩 준비해 무작위로 교환할 때, 아무도 자기 선물을 다시 받지 않는 경우의 수를 셀 때 등장한다.

# 4. 중복조합 (Combination with replacement)

$n$가지 종류에서 **중복을 허용**해 $k$개를 고르는 경우의 수:
$$\binom{n+k-1}{k}$$

>[!tip] Stars and bars
>중복조합은 "별과 막대(stars and bars)" 그림으로 이해하면 직관적이다. $k$개의 별(선택한 개수)과 $n-1$개의 막대(종류를 구분하는 경계)를 한 줄로 배열한다고 생각하면, 전체 $n+k-1$개의 자리 중 별이 놓일 $k$개의 자리를 고르는 문제로 바뀐다.
>
>같은 아이디어로 "음이 아닌 정수 $n$을 $k$개의 음이 아닌 정수의 합으로 나타내는 방법의 수"도 $n$개의 별 사이(또는 양옆)에 $k-1$개의 막대를 놓는 문제로 환원된다.

# 5. 포함배제 원리 (Inclusion-exclusion)

두 사건/집합의 경우 [[Chapter 3. Elements of probability|Chapter 3]]의 Proposition 2에서 이미 증명한 것과 같다.
$$P(A\cup B)=P(A)+P(B)-P(A\cap B)$$

세 개로 확장하면,
$$P(A\cup B\cup C)=P(A)+P(B)+P(C)-P(A\cap B)-P(A\cap C)-P(B\cap C)+P(A\cap B\cap C)$$

확률이 아니라 개수(카디널리티)로 써도 똑같은 형태다.
$$|A\cup B\cup C|=|A|+|B|+|C|-|A\cap B|-|A\cap C|-|B\cap C|+|A\cap B\cap C|$$

패턴: 홀수 개의 교집합은 더하고, 짝수 개의 교집합은 뺀다.

# 한 줄 요약
> 조합 = 순열 / $k!$. 중복조합은 stars and bars로 그림을 그려서 풀고, 여러 집합의 합집합 크기는 포함배제 원리(홀수 개 교집합은 +, 짝수 개는 -)로 구한다.
