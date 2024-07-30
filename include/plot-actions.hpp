#pragma once

#include "alloc-action.hpp"

#include <string>
#include <vector>

struct PlotOptions {

  enum Plotformat : uint8_t { Console, Svg, Obj };

  enum PlotScale : uint8_t { Linear, Log, Sqrt };

  enum PlotIndicate : uint8_t { Thread, Caller };

  enum PlotLayout : uint8_t { Timeline, Address };

  Plotformat format{Plotformat::Svg};
  std::string path{""};
  PlotScale height_scale{PlotScale::Sqrt};
  PlotIndicate z_indicates{PlotIndicate::Thread};

  PlotLayout layout{PlotLayout::Timeline};

  bool show_text{true};
  size_t text_max_height{24};
  double text_height_fraction{0.4};

  bool filter_cpp{true};
  bool filter_c{true};
  bool filter_cuda{true};

  size_t svg_margin = 420;
  size_t svg_width = 2000;
  size_t svg_height = 1460;
};

void mem_check_plot_alloc_actions(std::vector<AllocAction> actions);