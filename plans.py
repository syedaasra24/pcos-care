"""Diet and exercise plan generators for PCOS Care.

General lifestyle guidance only (no calorie targets, no supplements, no
medical claims). Plans use BMI category, food preference / fitness level,
and the user's latest PCOS screening result.
"""


def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight", "Low"
    if bmi < 25:
        return "Healthy weight", "Low"
    if bmi < 30:
        return "Overweight", "Moderate"
    return "Obese range", "High"          # second value is kept for the DB column


# ---------------------------------------------------------------- DIET

DIET_BASE = {
    "Vegetarian": {
        "breakfast": "Moong dal chilla or vegetable besan cheela with mint chutney + 1 glass milk or curd",
        "lunch": "2 multigrain rotis (or 1 small bowl brown rice) + dal + seasonal sabzi + cucumber-tomato salad + curd",
        "snacks": "Roasted chana, a handful of almonds/walnuts, or fruit with a few seeds",
        "dinner": "Paneer or tofu bhurji + 1-2 rotis + sauteed vegetables or vegetable soup",
    },
    "Eggetarian": {
        "breakfast": "2 egg vegetable omelette + 1 slice multigrain toast, or vegetable oats upma",
        "lunch": "2 multigrain rotis (or 1 small bowl brown rice) + dal + sabzi + salad + curd",
        "snacks": "Boiled egg, roasted chana, or a handful of nuts with fruit",
        "dinner": "Egg curry or paneer + 1-2 rotis + sauteed vegetables or soup",
    },
    "Non-vegetarian": {
        "breakfast": "Vegetable omelette or boiled eggs + 1 slice multigrain toast, or oats with nuts",
        "lunch": "Grilled fish or chicken curry (light oil) + 2 rotis or small bowl brown rice + salad",
        "snacks": "Roasted chana, nuts, or fruit with a few seeds",
        "dinner": "Grilled chicken, fish or paneer + sauteed vegetables + 1 roti or soup",
    },
    "Vegan": {
        "breakfast": "Moong dal chilla or vegetable oats upma + 1 glass soy or almond milk",
        "lunch": "2 multigrain rotis (or 1 small bowl brown rice) + dal or rajma + sabzi + salad",
        "snacks": "Roasted chana, a handful of nuts/seeds, or fruit with peanut butter",
        "dinner": "Tofu or chickpea curry + 1-2 rotis + sauteed vegetables or soup",
    },
}

DIET_BY_CATEGORY = {
    "Underweight": [
        "Eat 5-6 smaller meals a day and never skip breakfast.",
        "Add healthy energy-dense foods: nuts, seeds, nut butter, avocado, banana, and a little ghee or olive oil.",
        "Include protein at every meal. Some strength training will help build muscle, not just weight.",
    ],
    "Healthy weight": [
        "Keep meals balanced: half the plate vegetables, a quarter protein, a quarter whole grains.",
        "Aim for regular meal times; long gaps can cause energy crashes and cravings.",
    ],
    "Overweight": [
        "Swap refined carbs (white rice, maida, sugar) for whole grains, millets and dals.",
        "Fill half your plate with vegetables and keep portions of rice/roti moderate.",
        "Choose fruit instead of juices and sweets.",
    ],
    "Obese range": [
        "Swap refined carbs (white rice, maida, sugar) for whole grains, millets and dals.",
        "Fill half your plate with vegetables and have protein in every meal to stay full longer.",
        "Avoid sugary drinks, fried snacks and bakery items. Small, steady changes work better than strict diets.",
        "Ask your doctor or a dietitian for a personal plan before making big changes.",
    ],
}

DIET_GENERAL = [
    "Drink 2-3 litres of water daily.",
    "Prefer high-fibre, high-protein, low-glycaemic foods (dals, millets, oats, vegetables).",
    "Limit sugary drinks, packaged snacks and deep-fried food.",
    "Include healthy fats: nuts, seeds, olive/groundnut oil, and omega-3 sources such as flaxseed or walnuts (fish if you eat it).",
]

INSULIN_NOTE = (
    "Since you mentioned insulin resistance or diabetes: spread carbohydrates evenly across meals, "
    "pair them with protein or fibre, and follow your doctor's advice on diet and medicines."
)

HIGH_RISK_NOTE = (
    "Your latest screening showed a higher PCOS risk. This plan favours steady blood-sugar, high-fibre eating, "
    "which many people with PCOS find helpful. A screening is not a diagnosis, so please see a gynecologist to confirm."
)


def build_diet_plan(bmi, preference="Vegetarian", insulin_issue=False, high_risk=False):
    if preference not in DIET_BASE:
        preference = "Vegetarian"

    category, db_level = bmi_category(bmi)
    base = DIET_BASE[preference]

    notes = list(DIET_BY_CATEGORY[category])
    if insulin_issue:
        notes.insert(0, INSULIN_NOTE)
    if high_risk:
        notes.insert(0, HIGH_RISK_NOTE)
    notes += DIET_GENERAL

    return {
        "category": category,
        "db_level": db_level,
        "preference": preference,
        "breakfast": base["breakfast"],
        "lunch": base["lunch"],
        "snacks": base["snacks"],
        "dinner": base["dinner"],
        "notes": notes,
    }


# ------------------------------------------------------------ EXERCISE

LEVELS = ("Beginner", "Moderate", "Active")


def build_exercise_plan(bmi, level="Beginner", joint_issue=False, high_risk=False, irregular_cycle=False):
    if level not in LEVELS:
        level = "Beginner"

    category, db_level = bmi_category(bmi)

    # Low-impact only for higher BMI or joint problems
    low_impact = joint_issue or category == "Obese range"

    if level == "Beginner":
        warmup = "5 min easy walking + gentle arm and leg stretches"
        cardio = "20 min brisk walking" if low_impact else "20-25 min brisk walking"
        strength = "2 days/week: bodyweight squats, wall push-ups, glute bridges (2 sets x 10)"
        duration = "30 minutes"
    elif level == "Moderate":
        warmup = "5-8 min brisk walk + dynamic stretches"
        cardio = ("30 min brisk walking / cycling / swimming" if low_impact
                  else "30 min brisk walking, cycling, swimming or light jogging")
        strength = "2-3 days/week: squats, lunges, band or dumbbell rows, planks (3 sets x 10-12)"
        duration = "40 minutes"
    else:
        warmup = "8-10 min jog or skipping + dynamic stretches"
        cardio = ("35-40 min cycling / swimming / cross-trainer" if low_impact
                  else "30-40 min jogging, cycling, dance or HIIT (2 days max)")
        strength = "3 days/week: weights or resistance bands for legs, back, chest and core (3 sets x 10-12)"
        duration = "50 minutes"

    if category == "Underweight":
        cardio = "15-20 min light walking or cycling (don't overdo cardio)"
        strength = "3 days/week: strength training with progressive weights to build muscle"

    yoga = "Cat-Cow, Cobra, Butterfly (Baddha Konasana), Child's pose + 5 min slow breathing"
    if high_risk or irregular_cycle:
        yoga = "Butterfly (Baddha Konasana), Cat-Cow, Cobra, Bridge pose, Child's pose + 5-10 min deep breathing for stress"

    notes = [
        "Aim for about 150 minutes of moderate activity a week, plus 2 strength days.",
        "Strength training helps the body use insulin better, which is useful in PCOS.",
        "Start gently and increase time or intensity slowly, over weeks.",
        "Stop and rest if you feel pain, dizziness or breathlessness.",
        "Sleep 7-8 hours and manage stress; both affect hormones.",
    ]
    if low_impact:
        notes.insert(0, "Your plan is low-impact to protect your joints. Avoid jumping and running for now.")
    if high_risk:
        notes.insert(0, "Your latest screening showed a higher PCOS risk. Regular exercise helps, but please also see a gynecologist.")
    notes.append("Consult your doctor before starting a new routine, especially if you have any health condition.")

    return {
        "category": category,
        "db_level": db_level,
        "level": level,
        "warmup": warmup,
        "cardio": cardio,
        "strength": strength,
        "yoga": yoga,
        "duration": duration,
        "notes": notes,
    }