from random import randint

from sympy import isprime

primes = []
non_primes = []

while len(primes) < 100000:
    value = randint(1, 1 << 63) * 2 - 1

    if isprime(value):
        primes.append(value)

while len(non_primes) < 100000:
    value = randint(0, (1 << 64) - 1)

    if not isprime(value):
        non_primes.append(value)

with open('primes.csv', 'w') as file:
    file.write('integer,primality\n')

    for prime in primes:
        file.write(f'{prime},1\n')
    for non_prime in non_primes:
        file.write(f'{non_prime},0\n')
