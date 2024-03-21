# Creating own model 
from textblob import TextBlob
from language_tool_python import LanguageTool
class SpellCheckerModule:
    def __init__(self):
        self.spell_check = TextBlob("") # mpty string 
        self.grammar_check = LanguageTool('en-US')

    def correct_spell(self, text):
        words = text.split() # split the senetence into words
        corrected_words = []
        for word in words:
            corrected_word = str(TextBlob(word).correct())
            corrected_words.append(corrected_word)
        return " ".join(corrected_words)

    def correct_grammar(self, text):
        matches = self.grammar_check.check(text)

        found_mistakes = []
        for mistake in matches:
            found_mistakes.append(mistake.ruleId)
        found_mistakes_count = len(found_mistakes)
        return found_mistakes, found_mistakes_count
''' if __name__ == "__main__":
    ob = SpellCheckerModule()
    msg = "Welcomee to CBIT collage. I am Triveni from CSE - 1 thrad yaer"
    print(ob.correct_spell(msg))'''