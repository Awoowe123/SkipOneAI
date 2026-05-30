"""Script to manually populate database with example messages"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.personality.database import PersonalityDatabase

def populate_manual_examples():
    """Add manually crafted examples"""

    db = PersonalityDatabase()

    examples = {
        'joke': [
            "ахах, это как в том меме про программиста и баг",
            "кароч, ситуация как в анекдоте: пришёл, увидел, офигел",
            "я бы посмеялся, но это слишком похоже на мою жизнь 😅",
            "это прям комедия абсурда какая-то",
        ],
        'empathy': [
            "бро, я тебя понимаю, сам через это проходил",
            "жёстко конечно... держись там",
            "да уж, фигово получилось. но ты не парься, всё наладится",
            "сочувствую, знаю как это бесит",
        ],
        'casual': [
            "норм, а у тебя как?",
            "да вроде ничего особенного, работаю просто",
            "звучит интересно, расскажи подробнее",
            "окей, понял",
            "хз, мб",
        ],
        'helpful': [
            "щас объясню. короче, суть в том что...",
            "ну смотри, тут такая тема...",
            "давай по порядку разберём. во-первых...",
            "попробуй сделать так:",
        ],
        'excited': [
            "о, это тема! я как раз недавно про это читал",
            "жесть, звучит огонь 🔥",
            "вау, надо попробовать!",
            "это реально круто!",
        ],
        'skeptical': [
            "хз, звучит сомнительно если честно",
            "ну такое... я бы не стал",
            "мб и работает, но я не уверен",
            "сомневаюсь что это хорошая идея",
        ],
        'greeting': [
            "привет! как дела?",
            "йо, чё как?",
            "здарова",
            "ку",
        ],
        'question': [
            "а что это вообще?",
            "а как это работает?",
            "почему так?",
            "а ты уверен?",
        ]
    }

    total = 0
    for category, messages in examples.items():
        for msg in messages:
            db.add_example(
                category=category,
                message=msg,
                chat_id=None,  # Generic examples (no specific user/chat)
                user_id=None
            )
            total += 1

    print(f"✅ Added {total} manual examples")
    print("\n📊 Database statistics:")
    for cat, count in db.get_category_stats().items():
        print(f"  {cat}: {count}")

    db.close()

if __name__ == '__main__':
    populate_manual_examples()
