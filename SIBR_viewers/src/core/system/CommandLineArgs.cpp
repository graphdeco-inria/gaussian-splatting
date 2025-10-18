/*
 * Copyright (C) 2020, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr
 */



#include "core/system/CommandLineArgs.hpp"
#include "core/system/Utils.hpp"


namespace sibr
{
	CommandLineArgs CommandLineArgs::global = CommandLineArgs();

	const CommandLineArgs & getCommandLineArgs()
	{
		return CommandLineArgs::getGlobal();
	}

	CommandLineArgs & CommandLineArgs::getGlobal()
	{
		static bool first = true;
		if (!global.init && first) {
			SIBR_WRG << "CommandLineArgs::parseMainArgs(ac, av) was not called rigth after main(ac, av) \n default value (empty command line) will be used" << std::endl;
			first = false;
		}
		return global;
	}

	void CommandLineArgs::parseMainArgs(const int argc, const char * const * argv)
	{
		static const std::vector<std::string> acceptable_prefixes = { "--", "-" };

		global.args.clear();

		global.args["app_path"] = { std::string(argv[0])};

		std::string current_arg;
		for (int i = 1; i < argc; ++i) {
			std::string arg = std::string(argv[i]);
			bool new_arg = false;
			for (const auto & prefix : acceptable_prefixes) {
				if (arg.substr(0, prefix.size()) == prefix) {
					current_arg = arg.substr(prefix.size());
					new_arg = true;		
					break;
				}
			}
			if (current_arg.empty()) {
				continue;
			}
			if (new_arg) {
				if (global.args.count(current_arg) > 0) {
					SIBR_WRG << "Collision for argument : " << arg << std::endl;
				} else {
					global.args[current_arg] = {};
				}				
			} else {
				global.args[current_arg].push_back(arg);
			}
		}

		global.init = true;
	}

	bool CommandLineArgs::contains(const std::string & key) const
	{
		return args.count(key) > 0;
	}

	int CommandLineArgs::numArguments(const std::string & key) const
	{
		if (contains(key)) {
			return (int)args.at(key).size();
		} else {
			return -1;
		}
	}


	void CommandLineArgs::displayHelp() const {
		// Find the maximum length.
		size_t maxLength = 0;
		for (const auto & command : commands) {
			maxLength = std::max(maxLength, command.first.size());
		}

		const Path path = args.at("app_path")[0];
		SIBR_LOG << "Help for " << path.filename().string() << ":" << std::endl;
		for(const auto & command : commands) {
			// Pad to align everything.
#ifdef WIN32 // green
			std::string req = "[required]";
			std::string sec = command.second, xx;
			bool tgreen = false;
			if(sec.substr(sec.size()-req.size(), req.size()+1) == req) {
				setupConsole();
				printf("\x1b[32m");
				tgreen = true;
			}
#endif
			std::cout << "\t" << "--" << command.first;
			std::cout << std::string(int(maxLength) - command.first.size() + 1, ' ');
			std::cout << command.second << std::endl;

#ifdef WIN32
			if( tgreen )
				restoreConsole();
#endif
		}
		std::cout << std::endl;
	}
	
	void CommandLineArgs::registerCommand(const std::string & key, const std::string & description, const std::string & defaultValue) {
		// Register the command.
		std::string defaultDesc = description.empty() ? "" : " ";
		defaultDesc.append("(default: " + defaultValue + ")");
		commands[key] = description + defaultDesc;
	}

	void CommandLineArgs::registerRequiredCommand(const std::string & key, const std::string & description) {
		// Register the command.
		std::string defaultDesc = description.empty() ? "" : " ";
		defaultDesc.append("[required]");
		commands[key] = description + defaultDesc;
	}


	AppArgs::AppArgs()
	{
		Path path = CommandLineArgs::getGlobal().getRequired<std::string>("app_path");
		appName = path.filename().string();
		appPath = path.parent_path().string();
	}

	void AppArgs::displayHelpIfRequired() const {
		if(showHelp.get()) {
			getCommandLineArgs().displayHelp();
		}
	}

} // namespace sirb

