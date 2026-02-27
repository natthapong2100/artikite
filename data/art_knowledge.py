"""
data/art_knowledge.py
──────────────────────
Art history knowledge corpus.
In a real project this would be loaded from PDFs, Wikipedia, books, etc.
Each entry is a "document" that will be chunked and stored in ChromaDB.

Structure:
    ART_DOCUMENTS = list of dicts with:
        - id       : unique identifier
        - title    : document title (stored as metadata)
        - category : movement / artist / period (metadata for filtering)
        - content  : the actual text to embed and retrieve
"""

ART_DOCUMENTS = [

    # ── RENAISSANCE ─────────────────────────────────────────────
    {
        "id": "renaissance_overview",
        "title": "The Italian Renaissance",
        "category": "movement",
        "content": """
        The Italian Renaissance (14th–17th century) was a cultural and artistic revolution
        originating in Florence, Italy. It marked the transition from the Middle Ages to
        modernity. The word 'Renaissance' means 'rebirth' — referring to the renewed
        interest in classical Greek and Roman art, philosophy, and literature.

        Key characteristics of Renaissance art include linear perspective (the mathematical
        technique for depicting depth on a flat surface), naturalism (representing subjects
        as they appear in real life), humanism (placing humans at the center of the universe),
        and chiaroscuro (the dramatic contrast of light and shadow).

        The Renaissance is typically divided into Early Renaissance (1400s), High Renaissance
        (1490s–1527), and Mannerism (late 16th century). Florence was the cradle under the
        patronage of the Medici family, who funded artists and scholars extensively.
        """
    },
    {
        "id": "leonardo_da_vinci",
        "title": "Leonardo da Vinci",
        "category": "artist",
        "content": """
        Leonardo da Vinci (1452–1519) is widely regarded as the ultimate Renaissance Man —
        a painter, sculptor, architect, scientist, and inventor. Born in Vinci, Tuscany,
        he trained under Andrea del Verrocchio in Florence before working for Ludovico Sforza
        in Milan and later moving to Rome and France.

        His most celebrated paintings include the Mona Lisa (c. 1503–1519), famous for its
        enigmatic smile and pioneering use of sfumato (smoky atmospheric blending), and
        The Last Supper (c. 1495–1498), a mural depicting the moment Jesus announces
        his betrayal, celebrated for its dramatic perspective and emotional expression.

        Leonardo's notebooks contain thousands of pages of scientific sketches — studies
        of anatomy, hydraulics, flight, and optics — that were centuries ahead of his time.
        He pioneered the technique of sfumato, where tones blend without visible transitions,
        creating a hazy, dreamlike quality.
        """
    },
    {
        "id": "michelangelo",
        "title": "Michelangelo Buonarroti",
        "category": "artist",
        "content": """
        Michelangelo (1475–1564) was a Florentine sculptor, painter, architect, and poet,
        considered one of the greatest artists of all time. He was a rival of Leonardo da Vinci
        and both worked under Medici patronage.

        His sculpture David (1501–1504) — a 17-foot marble statue of the biblical hero —
        became a symbol of Florentine civic pride and the ideal of human beauty. The Pietà
        (1498–1499) shows Mary holding the dead Christ and is remarkable for its emotional
        restraint and technical mastery.

        He painted the Sistine Chapel ceiling (1508–1512) for Pope Julius II, covering
        over 500 square meters with scenes from Genesis. The Creation of Adam, showing God
        extending his finger toward Adam, is among the most recognized images in Western art.
        Later, he painted The Last Judgement (1536–1541) on the altar wall.

        As an architect, he designed the dome of St. Peter's Basilica in Rome.
        """
    },
    {
        "id": "raphael",
        "title": "Raphael Sanzio",
        "category": "artist",
        "content": """
        Raphael (1483–1520) was an Italian painter and architect of the High Renaissance,
        born in Urbino. He is celebrated for the clarity, harmony, and grace of his compositions.
        He worked in Florence studying Leonardo and Michelangelo before being summoned to Rome
        by Pope Julius II.

        His most famous work is The School of Athens (1509–1511), a fresco in the Vatican's
        Apostolic Palace depicting great philosophers of antiquity including Plato and Aristotle
        in an idealized architectural setting. It represents the synthesis of Christian and
        classical thought.

        Raphael was enormously prolific and ran a large workshop. His Madonnas — idealized
        portrayals of the Virgin Mary — were hugely influential. He died young at 37,
        leaving an enormous legacy that influenced Western painting for centuries.
        """
    },

    # ── BAROQUE ─────────────────────────────────────────────────
    {
        "id": "baroque_overview",
        "title": "The Baroque Period",
        "category": "movement",
        "content": """
        The Baroque period (1600–1750) emerged in Rome as a response to the Protestant
        Reformation. The Catholic Church commissioned dramatic, emotional art to inspire
        religious devotion. Baroque art is characterized by grandeur, exaggeration of motion,
        dramatic use of light (tenebrism), and emotional intensity.

        Key characteristics include: strong contrast between light and darkness (chiaroscuro
        taken to extremes), dynamic compositions with diagonal lines and movement, rich colors,
        and an appeal to the senses and emotions rather than intellect.

        Baroque spread across Europe and evolved differently in each country. In Italy:
        Caravaggio and Bernini. In Flanders: Rubens. In the Netherlands: Rembrandt and Vermeer.
        In Spain: Velázquez. In France: Poussin and later the Rococo reaction.
        """
    },
    {
        "id": "caravaggio",
        "title": "Caravaggio",
        "category": "artist",
        "content": """
        Michelangelo Merisi da Caravaggio (1571–1610) was an Italian Baroque painter whose
        revolutionary style transformed European art. He pioneered tenebrism — extreme
        chiaroscuro where figures dramatically emerge from very dark backgrounds.

        His paintings feature realistic, sometimes gritty depictions of religious scenes
        using common people as models. Notable works include The Calling of Saint Matthew
        (1600), where a beam of light illuminates a tax collector called by Christ, and
        Judith Beheading Holofernes (1598–1599), showing brutal realism rarely seen in
        religious art.

        Caravaggio's life was turbulent — he fled Rome in 1606 after killing a man in a
        brawl. Despite this, his influence on Baroque painters across Europe was enormous.
        Artists who adopted his style were called 'Caravaggisti'.
        """
    },
    {
        "id": "rembrandt",
        "title": "Rembrandt van Rijn",
        "category": "artist",
        "content": """
        Rembrandt van Rijn (1606–1669) is considered the greatest Dutch Golden Age painter.
        He was a master of light, shadow, and psychological depth. Born in Leiden, he worked
        mainly in Amsterdam and produced roughly 300 paintings, 300 etchings, and 2,000 drawings.

        His masterpiece The Night Watch (1642) is a large-scale group portrait of a militia
        company. It was revolutionary for its dramatic composition, use of light and shadow,
        and sense of movement — unlike the static group portraits typical of the time.

        Rembrandt painted over 80 self-portraits across his lifetime, creating an unparalleled
        record of aging and self-examination. His later works show increasingly loose brushwork
        and profound psychological depth. He died in poverty despite earlier success.
        """
    },

    # ── IMPRESSIONISM ────────────────────────────────────────────
    {
        "id": "impressionism_overview",
        "title": "Impressionism",
        "category": "movement",
        "content": """
        Impressionism originated in France in the 1860s–1880s as a radical break from
        academic painting traditions. The name came mockingly from Claude Monet's painting
        Impression, Sunrise (1872), which a critic dismissed as a mere 'impression.'

        Key characteristics: painting outdoors (en plein air) to capture natural light,
        visible and loose brushstrokes, focus on light and its changing qualities, ordinary
        subject matter (cafés, gardens, leisure), and capturing a fleeting moment rather
        than a constructed scene.

        Impressionists rejected the dark studio paintings of the past and the stiff poses
        of academic art. They were initially rejected by the official Salon de Paris and
        held their own independent exhibitions starting in 1874.

        Major Impressionists include Claude Monet, Pierre-Auguste Renoir, Edgar Degas,
        Berthe Morisot, Camille Pissarro, and Alfred Sisley. Post-Impressionists who
        built on the movement include Cézanne, Van Gogh, Gauguin, and Seurat.
        """
    },
    {
        "id": "claude_monet",
        "title": "Claude Monet",
        "category": "artist",
        "content": """
        Claude Monet (1840–1926) is the most emblematic Impressionist painter, known for
        his obsessive study of light and its effects. Born in Paris and raised in Normandy,
        he spent decades painting the same subjects under different lighting conditions.

        His Water Lilies series (approximately 250 paintings, 1896–1926) depicts the pond
        at his garden in Giverny, Normandy, at different times of day and seasons. The late
        large-scale panels in the Orangerie in Paris are considered masterpieces of modern
        abstraction — decades before Abstract Expressionism.

        Monet's series paintings also include Haystacks (1890–1891), Rouen Cathedral
        (1892–1894), and the Thames in London. As his eyesight deteriorated from cataracts,
        his palette became more intense and his forms less defined — some see this as even
        more expressive work.
        """
    },
    {
        "id": "van_gogh",
        "title": "Vincent van Gogh",
        "category": "artist",
        "content": """
        Vincent van Gogh (1853–1890) was a Dutch Post-Impressionist painter whose work
        profoundly influenced 20th century art. He produced over 2,100 works in just 10 years
        before his death at 37. He sold only one painting in his lifetime.

        Van Gogh's style is characterized by bold, swirling brushstrokes, vivid colors,
        and emotional intensity. He painted The Starry Night (1889) while in the Saint-Paul
        asylum in Saint-Rémy-de-Provence, depicting a swirling night sky over a village.

        Other famous works include Sunflowers (1888), his series of self-portraits, and
        The Bedroom in Arles (1888). He suffered from severe mental illness throughout his
        life and famously cut off part of his ear during a breakdown in 1888. He died from
        a gunshot wound in Auvers-sur-Oise, widely believed to be suicide.

        His letters to his brother Theo provide extraordinary insight into his artistic
        vision and emotional life.
        """
    },

    # ── MODERN ART ───────────────────────────────────────────────
    {
        "id": "cubism_overview",
        "title": "Cubism",
        "category": "movement",
        "content": """
        Cubism was developed in Paris between 1907 and 1914 primarily by Pablo Picasso and
        Georges Braque, influenced by Paul Cézanne's geometric simplifications. It was the
        most radical and influential avant-garde movement of the early 20th century.

        Cubism broke objects into geometric fragments and showed multiple viewpoints
        simultaneously on a single flat surface, abandoning traditional perspective.
        Analytic Cubism (1908–1912) used muted earthy colors with fragmented forms.
        Synthetic Cubism (after 1912) introduced collage elements and brighter colors.

        Cubism fundamentally challenged the Western tradition of representing three-dimensional
        space on a flat surface that had prevailed since the Renaissance. It influenced
        Futurism, Constructivism, Expressionism, and eventually Abstract Expressionism.
        """
    },
    {
        "id": "pablo_picasso",
        "title": "Pablo Picasso",
        "category": "artist",
        "content": """
        Pablo Picasso (1881–1973) is considered the most influential artist of the 20th century.
        Born in Málaga, Spain, he spent most of his adult life in France. His career spanned
        over 80 years and he worked across painting, sculpture, printmaking, and ceramics.

        His early career had distinct periods: the Blue Period (1901–1904) with somber blue
        tones depicting poverty and isolation, and the Rose Period (1904–1906) with warmer
        colors depicting circus performers. With Les Demoiselles d'Avignon (1907), he shattered
        traditional representation and pioneered Cubism alongside Georges Braque.

        Guernica (1937) is his most politically charged work — a large black-and-white painting
        responding to the Nazi bombing of the Basque town of Guernica during the Spanish Civil
        War. It is a powerful anti-war statement and one of the most famous paintings of the
        20th century.
        """
    },
    {
        "id": "surrealism_overview",
        "title": "Surrealism",
        "category": "movement",
        "content": """
        Surrealism was a cultural movement founded in Paris in 1924 by poet André Breton,
        influenced by Sigmund Freud's theories of the unconscious mind and dream symbolism.
        It sought to unlock the unconscious mind and merge dream and reality.

        Visual techniques included automatism (spontaneous, unplanned creation), dreamlike
        imagery combining unrelated objects, hyper-realistic depictions of irrational scenes,
        and the distortion of familiar objects.

        Major Surrealist painters include Salvador Dalí (The Persistence of Memory, 1931),
        René Magritte (The Treachery of Images — 'Ceci n'est pas une pipe', 1929),
        Max Ernst, Joan Miró, and Frida Kahlo (though she resisted the label).

        Surrealism influenced literature, film, advertising, and later Pop Art. It flourished
        between World War I and World War II, with many artists fleeing to New York as
        World War II approached, cross-pollinating American Abstract Expressionism.
        """
    },
    {
        "id": "frida_kahlo",
        "title": "Frida Kahlo",
        "category": "artist",
        "content": """
        Frida Kahlo (1907–1954) was a Mexican painter known for her intensely personal and
        symbolic self-portraits. She suffered a near-fatal bus accident at 18 that left her
        with lifelong physical pain, and she began painting during her long recovery.

        Her work is deeply autobiographical, exploring themes of identity, postcolonialism,
        gender, Mexican folklore, and physical pain. Of her 143 paintings, 55 are self-portraits.
        She famously said: 'I paint my own reality.'

        The Two Fridas (1939) shows two versions of herself with exposed hearts — one European,
        one Mexican — painted after her divorce from muralist Diego Rivera. Despite André Breton
        labeling her a Surrealist, she rejected this, saying her work depicted her own reality
        not her dreams.

        She and Diego Rivera were central figures in the Mexican muralist movement and
        Mexican national identity. Her home, La Casa Azul (The Blue House) in Coyoacán,
        is now a museum.
        """
    },
]


def get_all_documents() -> list[dict]:
    """Return the full art history corpus."""
    return ART_DOCUMENTS


def get_documents_by_category(category: str) -> list[dict]:
    """Filter documents by category: 'artist' or 'movement'."""
    return [doc for doc in ART_DOCUMENTS if doc["category"] == category]
