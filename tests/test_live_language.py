import pytest
from app.live import AFFIRM, NEG, foreign_language


@pytest.mark.parametrize("text", [
    "Hola. ¿En qué puedo ayudarte hoy?", "Guten Tag, wie geht's?", "Bonjour, ça va?", "Merhaba, nasılsınız?", "مرحبا",
])
def test_foreign_replies_are_flagged(text):
    assert foreign_language(text)


@pytest.mark.parametrize("text", [
    "გამარჯობა, თმის შეჭრა 30 ლარია.", "მარტო ქართულად მესმის.", "Здравствуйте! Чем могу помочь?",
    "Sure, a haircut is thirty lari. Anything else?", "Your booking is at five o'clock, it's Giorgi.",
])
def test_allowed_languages_pass(text):
    assert not foreign_language(text)


def test_yes_no_vocabulary_has_no_overlap():
    assert not (AFFIRM & NEG)
