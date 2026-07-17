"""Sample traces for the RAG retrieval-explanation dashboard."""

from __future__ import annotations

SAMPLE_TRACES = {
    "Marie Curie Nobel categories": {
        "page": "Page 1 - Puzzle Pieces",
        "question": "Which two Nobel Prize categories did Marie Curie win?",
        "target_answer": "Marie Curie won Nobel Prizes in Physics and Chemistry.",
        "takeaway": (
            "The answer needs two complementary evidence chunks: one for Physics and one "
            "for Chemistry. This is the cleanest scenario for explaining chunk interactions."
        ),
        "chunks": [
            {
                "title": "1903 Nobel Prize in Physics",
                "text": (
                    "Marie Curie shared the 1903 Nobel Prize in Physics with Pierre Curie "
                    "and Henri Becquerel for research on radiation phenomena. The award "
                    "establishes Physics as one of her Nobel Prize categories."
                ),
                "evidence_role": "Physics evidence",
            },
            {
                "title": "1911 Nobel Prize in Chemistry",
                "text": (
                    "In 1911, Marie Curie received the Nobel Prize in Chemistry for the "
                    "discovery of radium and polonium. This second award establishes "
                    "Chemistry as the other category in the answer."
                ),
                "evidence_role": "Chemistry evidence",
            },
            {
                "title": "Curie family background",
                "text": (
                    "Pierre Curie was a physicist and Marie Curie's husband. Their family "
                    "became closely associated with radioactivity research. This background "
                    "does not list both Nobel Prize categories."
                ),
                "evidence_role": "Background — family context",
            },
            {
                "title": "Nobel Peace Prize note",
                "text": (
                    "The Nobel Peace Prize is awarded for work toward peace and diplomacy. "
                    "It is separate from the scientific Nobel Prize categories."
                ),
                "evidence_role": "Distractor — wrong category",
            },
        ],
    },
    "2008 Beijing Olympics host city": {
        "page": "Page 2 - Signal vs. Distractors",
        "question": "Which city hosted the 2008 Summer Olympics?",
        "target_answer": "Beijing hosted the 2008 Summer Olympics.",
        "takeaway": (
            "The first chunk is direct evidence. The other chunks mention Olympics, China, "
            "or later host cities, but they should rank lower."
        ),
        "chunks": [
            {
                "title": "2008 Summer Olympics",
                "text": (
                    "The 2008 Summer Olympics, officially the Games of the XXIX Olympiad, "
                    "were hosted by Beijing, China. This sentence directly answers the host "
                    "city question."
                ),
                "evidence_role": "Direct answer",
            },
            {
                "title": "Opening ceremony",
                "text": (
                    "The opening ceremony of the Beijing 2008 Olympics was held at the "
                    "National Stadium, also known as the Bird's Nest. It identifies a venue "
                    "inside Beijing but does not by itself state the host-city fact."
                ),
                "evidence_role": "Background — venue only",
            },
            {
                "title": "London 2012",
                "text": (
                    "London hosted the 2012 Summer Olympics, four years after the 2008 "
                    "Games and before Rio de Janeiro 2016. This chunk is a distractor because "
                    "it discusses another host city and another Olympic year."
                ),
                "evidence_role": "Distractor — wrong year",
            },
            {
                "title": "Olympic host bidding",
                "text": (
                    "Several cities have competed to host the Summer Olympics, including "
                    "Beijing, Toronto, Paris, Istanbul, and Osaka."
                ),
                "evidence_role": "Background — multiple cities",
            },
        ],
    },
    "Unsupported Eiffel Tower answer": {
        "page": "Page 3 - Missing Evidence",
        "question": "Who designed the Eiffel Tower?",
        "target_answer": "Gustave Eiffel's company designed the Eiffel Tower.",
        "takeaway": (
            "The retrieved chunks talk around Paris landmarks but do not contain the key "
            "supporting fact. The full-context support score should stay low."
        ),
        "chunks": [
            {
                "title": "Paris tourism",
                "text": (
                    "Paris is known for museums, historic boulevards, cafes, and major "
                    "landmarks visited by international travelers. This does not identify "
                    "the Eiffel Tower designer."
                ),
                "evidence_role": "Background — city context",
            },
            {
                "title": "World fairs",
                "text": (
                    "World fairs in the nineteenth century showcased industrial design, "
                    "architecture, engineering, and national pavilions."
                ),
                "evidence_role": "Background — era context",
            },
            {
                "title": "Seine river",
                "text": (
                    "The Seine flows through Paris and passes near several monuments, "
                    "bridges, and cultural institutions."
                ),
                "evidence_role": "Background — geography",
            },
            {
                "title": "French architecture",
                "text": (
                    "French architecture includes Gothic cathedrals, classical palaces, "
                    "Haussmann-era buildings, and modern landmarks."
                ),
                "evidence_role": "Background — architecture",
            },
        ],
    },
    "Australia capital confusion": {
        "page": "Page 4 - Conflicting Context",
        "question": "What is the capital city of Australia?",
        "target_answer": "Canberra is the capital city of Australia.",
        "takeaway": (
            "One chunk gives the correct answer, while other chunks mention Sydney and "
            "Melbourne, common distractors in this question."
        ),
        "chunks": [
            {
                "title": "Capital fact",
                "text": (
                    "Canberra is the capital city of Australia and the seat of the federal "
                    "government. This sentence directly supports the target answer."
                ),
                "evidence_role": "Direct answer",
            },
            {
                "title": "Sydney distractor",
                "text": (
                    "Sydney is Australia's largest city and is famous for the Sydney Opera "
                    "House and harbour. It is a common distractor for the capital question."
                ),
                "evidence_role": "Distractor — largest city",
            },
            {
                "title": "Melbourne context",
                "text": (
                    "Melbourne is the capital of Victoria and was an important city in "
                    "Australia's political history."
                ),
                "evidence_role": "Distractor — former context",
            },
            {
                "title": "Australian states",
                "text": (
                    "Australia has six states and several territories, each with its own "
                    "capital city and local government."
                ),
                "evidence_role": "Background — states",
            },
        ],
    },
    "Atlantic Ocean redundancy": {
        "page": "Page 5 - Redundancy Detection",
        "question": "Which ocean borders South America to the east?",
        "target_answer": "The Atlantic Ocean borders South America to the east.",
        "takeaway": (
            "Chunks 1 and 2 are near-duplicates that both answer the question. Their "
            "individual Shapley values are moderate (credit is split), but their pairwise "
            "interaction is strongly negative — a signal that retrieval fetched redundant "
            "evidence. This is what k-SII adds beyond standard Shapley values."
        ),
        "chunks": [
            {
                "title": "Atlantic Ocean — eastern border",
                "text": (
                    "The Atlantic Ocean lies to the east of South America, bordering its "
                    "entire eastern coastline from Venezuela down to Argentina. This directly "
                    "answers which ocean borders South America to the east."
                ),
                "evidence_role": "Direct answer",
            },
            {
                "title": "South America — ocean borders",
                "text": (
                    "South America is flanked by the Atlantic Ocean on its eastern side and "
                    "the Pacific Ocean on its western side. The eastern border with the "
                    "Atlantic Ocean spans the full length of the continent."
                ),
                "evidence_role": "Redundant — duplicate answer",
            },
            {
                "title": "Pacific Ocean — western border",
                "text": (
                    "The Pacific Ocean forms the western boundary of South America, running "
                    "along the coasts of Colombia, Ecuador, Peru, and Chile. This is the "
                    "opposite side from the eastern ocean border."
                ),
                "evidence_role": "Distractor — wrong ocean",
            },
            {
                "title": "South American geography",
                "text": (
                    "South America is the fourth largest continent by area and home to the "
                    "Amazon rainforest and the Andes mountain range."
                ),
                "evidence_role": "Background — geography",
            },
        ],
    },
}
