import ternip

temporalTagger = ternip.recogniser()

output = temporalTagger.tag(['what', 'yes'])

print output