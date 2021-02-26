// main.cpp

// Unused import for window identification
// #include <windows.h>
// #include <windef.h>
// #include <winuser.h>

// c++ headers
#include <iostream>
#include <vector>

// Custom headers
#include <interaction_lib/InteractionLib.h>
#include <interaction_lib/misc/InteractionLibPtr.h>

int main()
{
    // create the interaction library
    IL::UniqueInteractionLibPtr intlib(IL::CreateInteractionLib(IL::FieldOfUse::Interactive));

    // Cannot find window name ~ can't progress to using window based stuff
    // retreive window size of browser
    // HWND hwnd = FindWindowA(NULL, );
    // RECT rect;
    // GetWindowRect(hwnd, &rect);
    // float width = rect.right - rect.left;
    // float height = rect.bottom - rect.top;

    // Set up window area

    const float windowWidth = 1920;
    const float windowHeight = 1080;
    const float offset = 0.0f;

    intlib->CoordinateTransformAddOrUpdateDisplayArea(windowWidth, windowHeight);
    intlib->CoordinateTransformSetOriginOffset(offset, offset);

    // TODO: begin working on splitting the screen into X interactor chunks
    // set numnber of interactor rows and columns
    // assuming they're all the same size
    const int columns = 3;
    const int rows = 2;
    const int count = columns * rows;
    const float boxWidth = windowWidth / columns;
    const float boxHeight = windowHeight / rows;

    // this struct is used to maintain a focus count
    struct Focus
    {
        IL::InteractorId id    = IL::EmptyInteractorId();
        size_t           count = 0;
    };
    std::vector<Focus> focusVec;

    // Begin setup of interactors
    intlib->BeginInteractorUpdates();

    // setup ids and rectangles that define the interactors we want to use
    int colCount = 0;
    int rowCount = 0;
    const float z = 0.0f;  // This is the depth off the screen (increases towards user)
    for (size_t i = 0; i < count; i++)
    {
        IL::InteractorId id = i;

        // rect = {top_left_point_hori, top_left_point_vert, width, height}
        IL::Rectangle rect = {colCount*boxWidth, rowCount*boxHeight, boxWidth, boxHeight};
        ++colCount;
        if (colCount == columns)
        {
            ++rowCount;
            colCount = 0;
        }
        
        // Add them to the interactor system
        intlib->AddOrUpdateInteractor(id, rect, z);

        Focus focus;
        focus.id = id;
        focusVec.push_back(focus);
    }

    // Commit changes
    intlib->CommitInteractorUpdates();

    /* Need by end:
    image name, area of interest on file, fixation start, length and end, gap between fixations, horizontal and vert pos, pupil diam, area of interest

    Next bit is confidence results
    */

    // subscribe to gaze focus events
    // print event data to std out when called and count the number of consecutive focus events
    intlib->SubscribeGazeFocusEvents([](IL::GazeFocusEvent evt, void* context)
    {
        std::vector<Focus>& focusVec = *static_cast<std::vector<Focus>*>(context);
        std::cout
            << "Interactor: " << evt.id
            << ", focused: " << std::boolalpha << evt.hasFocus
            << ", timestamp: " << evt.timestamp_us << " us"
            << "\n";

        if (evt.hasFocus)
        {
            ++focusVec[evt.id].count;
        }
    }, &focusVec);

    // setup and maintain device connection, wait for device data between events and
    // update interaction library to trigger all callbacks
    // stop after 3 consecutive focus events on the same interactor
    std::cout << "Starting interaction library update loop.\n";

    constexpr size_t max_focus_count = 3;

    while (focus.count < max_focus_count)
    {
        intlib->WaitAndUpdate();
    }

    std::cout << "Interactor " << focus.id << " got focused " << focus.count << " times\n";

    system("pause");
}
