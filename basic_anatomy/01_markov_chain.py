import random

def generate_markov_chain(text, n_grams=2):

    words = text.split()
    markov_chain = {}

    for i in range(len(words) - n_grams):
        n_gram = ' '.join(words[i:i+n_grams])
        next_word = words[i+n_grams]

        if n_gram not in markov_chain:
            markov_chain[n_gram] = []
        markov_chain[n_gram].append(next_word)

    return markov_chain

def generate_text(markov_chain, n_grams=2, length=100):

    seed = random.choice(list(markov_chain.keys()))
    current_text = seed.split()
    generated_text = seed

    for _ in range(length):
        try:
            next_word = random.choice(markov_chain[seed])
        except KeyError:
            # If the n-gram is not found, choose a random seed
            seed = random.choice(list(markov_chain.keys()))
            next_word = random.choice(markov_chain[seed])

        generated_text += " " + next_word
        current_text.append(next_word)
        current_text = current_text[1:]  # Shift the window
        seed = ' '.join(current_text)

    return generated_text

text = "This is an example text to demonstrate the Markov Chain. Markov Chains are a probabilistic model."

markov_model = generate_markov_chain(text, n_grams=2)
generated_text = generate_text(markov_model, n_grams=2, length=20)

print(generated_text)