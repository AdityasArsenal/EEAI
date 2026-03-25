class Config:
    # Input Columns
    TICKET_SUMMARY = 'Ticket Summary'
    INTERACTION_CONTENT = 'Interaction content'

    # Type Columns to test
    TYPE_COLS = ['y2', 'y3', 'y4']
    CLASS_COL = 'y2'
    GROUPED = 'y1'
    
    # Chained Multi-output Targets
    CHAINED_TARGETS = {
        'y2': ['y2'],
        'y2_y3': ['y2', 'y3'],
        'y2_y3_y4': ['y2', 'y3', 'y4']
    }