"""Sample RAG traces for the Streamlit retrieval-explanation demo."""

from __future__ import annotations

SCENARIO_PAGES = {
    "Page 1 - Puzzle Pieces": {
        "title": "Puzzle Pieces: Multi-Chunk Grounding",
        "goal": (
            "Show a case where no single retrieved chunk fully supports the answer. "
            "The interesting signal is the interaction between complementary chunks."
        ),
        "best_for": "k-SII / STII interaction view",
    },
    "Page 2 - Signal vs. Distractors": {
        "title": "Signal vs. Distractors: Retrieval Ranking",
        "goal": (
            "Show a case where one retrieved chunk directly answers the question while "
            "other chunks are only keyword-related or misleading."
        ),
        "best_for": "SV / first-order ranking",
    },
    "Page 3 - Missing Evidence": {
        "title": "Missing Evidence: Unsupported Answer",
        "goal": (
            "Show a case where the retrieved context does not really support the target "
            "answer. A good explanation should avoid pretending that weak evidence is grounded."
        ),
        "best_for": "full-context support score",
    },
    "Page 4 - Conflicting Context": {
        "title": "Conflicting Context: Correct Chunk vs. Distractor Chunk",
        "goal": (
            "Show a case where one chunk states the answer, while another related chunk can "
            "pull attention toward a common wrong answer."
        ),
        "best_for": "ranking plus negative/redundant pair effects",
    },
}


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
            },
            {
                "title": "1911 Nobel Prize in Chemistry",
                "text": (
                    "In 1911, Marie Curie received the Nobel Prize in Chemistry for the "
                    "discovery of radium and polonium. This second award establishes "
                    "Chemistry as the other category in the answer."
                ),
            },
            {
                "title": "Curie family background",
                "text": (
                    "Pierre Curie was a physicist and Marie Curie's husband. Their family "
                    "became closely associated with radioactivity research. This background "
                    "does not list both Nobel Prize categories."
                ),
            },
            {
                "title": "Nobel Peace Prize note",
                "text": (
                    "The Nobel Peace Prize is awarded for work toward peace and diplomacy. "
                    "It is separate from the scientific Nobel Prize categories."
                ),
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
            },
            {
                "title": "Opening ceremony",
                "text": (
                    "The opening ceremony of the Beijing 2008 Olympics was held at the "
                    "National Stadium, also known as the Bird's Nest. It identifies a venue "
                    "inside Beijing but does not by itself state the host-city fact."
                ),
            },
            {
                "title": "London 2012",
                "text": (
                    "London hosted the 2012 Summer Olympics, four years after the 2008 "
                    "Games and before Rio de Janeiro 2016. This chunk is a distractor because "
                    "it discusses another host city and another Olympic year."
                ),
            },
            {
                "title": "Olympic host bidding",
                "text": (
                    "Several cities have competed to host the Summer Olympics, including "
                    "Beijing, Toronto, Paris, Istanbul, and Osaka."
                ),
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
            },
            {
                "title": "World fairs",
                "text": (
                    "World fairs in the nineteenth century showcased industrial design, "
                    "architecture, engineering, and national pavilions."
                ),
            },
            {
                "title": "Seine river",
                "text": (
                    "The Seine flows through Paris and passes near several monuments, "
                    "bridges, and cultural institutions."
                ),
            },
            {
                "title": "French architecture",
                "text": (
                    "French architecture includes Gothic cathedrals, classical palaces, "
                    "Haussmann-era buildings, and modern landmarks."
                ),
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
            },
            {
                "title": "Sydney distractor",
                "text": (
                    "Sydney is Australia's largest city and is famous for the Sydney Opera "
                    "House and harbour. It is a common distractor for the capital question."
                ),
            },
            {
                "title": "Melbourne context",
                "text": (
                    "Melbourne is the capital of Victoria and was an important city in "
                    "Australia's political history."
                ),
            },
            {
                "title": "Australian states",
                "text": (
                    "Australia has six states and several territories, each with its own "
                    "capital city and local government."
                ),
            },
        ],
    },
}
