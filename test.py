import ternip
import tokenizer

temporalTagger = ternip.recogniser()
'''
output = temporalTagger.tag([[('This', 'DT', set()), ('is', ' ', set()), ('some', ' ', set()), ('annotated', ' ', set()),
                                           ('embedded', ' ', set()), ('this', ' ', set()), ('morning', ' ', set()),
                                           ('.', ' ', set())], [('This', 'DT', set()), ('is', ' ', set()), ('the', ' ', set()),
                                           ('second', ' ', set()), ('sentence', ' ', set()), ('.', ' ', set())]])

print output
'''


def temporalExtractor(input):
    inputList = []
    if len(input[0][0]) > 1:
        for sent in input:
            tempList = []
            for word in sent:
                tempList.append((word[0], word[1], set()))
            inputList.append(tempList)
    else:
        for sent in input:
            tempList = []
            words = tokenizer.tokenize(sent)
            for word in words:
                tempList.append((word, ' ', set()))
            inputList.append(tempList)

    output = temporalTagger.tag(inputList)

    outputList = []
    for sent in output:
        tempList = []
        for word in sent:
            if len(word[2]) > 0:
                tempList.append(word[0])
        outputList.append(tempList)

    return outputList


if __name__ == "__main__":
    sents1 = ['This is some annotated samples for the experiment tonight.', 'I have no plan for tomorrow morning.',
             'The game next monday would be awesome!']
    sents2 = [
        [('This', 'O'), ('is', 'V'), ('some', 'D'), ('annotated', 'A'), ('samples', 'N'), ('for', 'P'), ('the', 'D'),
         ('experiment', 'N'), ('tonight', 'N'), ('.', ',')],
        [('I', 'O'), ('have', 'V'), ('no', 'D'), ('plan', 'N'), ('for', 'P'), ('tomorrow', 'N'), ('morning', 'N'),
         ('.', ',')],
        [('The', 'D'), ('game', 'N'), ('next', 'A'), ('monday', '^'), ('would', 'V'), ('be', 'V'), ('awesome', 'A'),
         ('!', ',')]]
    output = temporalExtractor(sents2)
    print output
