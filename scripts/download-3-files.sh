#!/bin/bash

# This script runs on Linux and MaxOS and downloads all the selected files to the current working directory in up to 5 parallel download streams.
# Should a download be aborted just run the entire script again, as partial downloads will be resumed. 
# We've finally got our server-side software sorted out and now positively encourage the use of parallel downloads
# of single files from version 2020-OCT.

# connect / read timeout for wget / curl
export TIMEOUT_SECS=300
# how many times do we want to automatically resume an interrupted download?
export MAX_RETRIES=3
# after a timeout, before we retry, wait a bit. Maybe the servers were overloaded, or there was some scheduled downtime.
# with the default settings we have 15 minutes to bring the dataportal service back up.
export WAIT_SECS_BEFORE_RETRY=300
# the files to be downloaded
LIST=("
https://almascience.eso.org/dataPortal/member.uid___A001_X3819_X177._035.522-00.274__sci.spw25.cube.I.selfcal.pbcor.fits.tar
https://almascience.eso.org/dataPortal/member.uid___A001_X3819_X177._035.522-00.274__sci.spw25_27_29_31.cont.I.selfcal.pbcor.fits.tar
https://almascience.eso.org/dataPortal/member.uid___A001_X3819_X177._035.522-00.274__sci.spw27.cube.I.selfcal.pbcor.fits.tar
")

# If we terminate the script using CTRL-C during parallel downloads, the remainder of the script is executed, asking if
# the user wants to unpack tar files. Not very nice. Exit the whole script when the user hits CTRL-C.
trap "end_session && exit" INT

# quickly log in to the server (if required) and create a cookies file. This is because some users are using shared machines for long-running
# downloads where the credentials can be seen in plain-text in the task list.
function start_session {
  export authentication_status=0
  if [ "ashleybarnes" != "anonymous" ]; then
    # only prompt for the password again if we haven't already. What could be going on here? The servers
    # have been restarted and the local cookie file is no longer valid.
    if [ -z "${password}" ]; then
      echo ""
      echo -n "Please enter the password for ALMA account ashleybarnes: "
      read -s password
      echo ""
      export password
    fi

    # ICT-17647 curl v7.29.0 is very slow when the --speed-limit option is used. Use wget by default until ALMA 
    # uses a more recent version of RHEL.
    if command -v "wget" > /dev/null 2>&1; then
      login_command=(wget --quiet --delete-after --no-check-certificate --auth-no-challenge --keep-session-cookies --save-cookies alma-rh-cookie.txt "--http-user=ashleybarnes" "--http-password=${password}")
    elif command -v "curl" > /dev/null 2>&1; then
      login_command=(curl -s -k -o /dev/null -c alma-rh-cookie.txt --fail "-u" "ashleybarnes:${password}")
    fi
    # echo "${login_command[@]}" "https://almascience.eso.org/dataPortal/downloads/login"
    $("${login_command[@]}" "https://almascience.eso.org/dataPortal/downloads/login")
    authentication_status=$?
    if [ $authentication_status -eq 0 ]; then
      echo "        OK: credentials accepted."
    else
      echo "        ERROR: login failed. Error code is ${authentication_status}"
    fi
  fi
}
export -f start_session

# clean up the cookies file after downloading is complete
function end_session {
	# Included unset variable: password, for ticket : ICT-16332
	unset password
	rm -fr alma-rh-cookie.txt
}
export -f end_session

# download a single file.
# attempt the download up to N times
function dl {
  local file=$1
  local filename=$(basename $file)
  # the nth attempt to download a single file
  local attempt_num=0
  local download_status=-1

  if command -v "wget" > /dev/null 2>&1; then
    local tool_name="wget"
    local download_command=(wget -c -q -nv --no-check-certificate --auth-no-challenge --load-cookies alma-rh-cookie.txt --timeout=${TIMEOUT_SECS} --tries=1)
  elif command -v "curl" > /dev/null 2>&1; then
    local tool_name="curl"
    local download_command=(curl -C - -S -s -k -O -f -b alma-rh-cookie.txt --speed-limit 1 --speed-time ${TIMEOUT_SECS})
  fi

  # manually retry downloads. 
  # I know wget and curl can both do this, but curl (as of 10.04.2018) will not allow retry and resume. I want consistent behaviour so 
  # we implement the retry mechanism ourselves.
  echo "starting download of $filename"
  until [ ${download_status} -ge 0 ]; do
    # echo "${download_command[@]}" "$file"
    $("${download_command[@]}" "$file")
    status=$?
    # echo "status ${status}"
    if [ ${status} -eq 0 ]; then
      echo "        successfully downloaded $filename"
      download_status=0
    elif [ $( df . | tail -1 | awk '{ print $4 }' ) == "0" ]; then
      echo "        ERROR downloading $filename: the disk is full. Please free space and re-run this script."
      # 255 stops xargs from executing the remaining the download tasts
      download_status=255
    else
      # users requested a string instead of a simple exit code - attempt to make it look like the curl output
      if [ ${status} -eq 8 ] && [ ${tool_name} = "wget" ]; then 
        echo "wget: (8) Server issued an error response."
      elif [ ${status} -eq 4 ] && [ ${tool_name} = "wget" ]; then
        echo "wget: (4) Network failure."
      elif { [ ${status} -eq 6 ] && [ ${tool_name} = "wget" ]; } || { [ ${status} -eq 22 ] && [ ${tool_name} = "curl" ]; }; then
        # if we restart the servers then our cookies are invalidated
        end_session
        echo "authentication error - retrying the login"
        start_session
        if [ $authentication_status -eq 0 ]; then
          # our retry attempt was hampered by a credentials issue. We shouldn't count this as an attempt. Give a bonus try...
          echo "        resuming download of $filename, still attempt $[${attempt_num}+1]"
          $("${download_command[@]}" "$file")
        fi
      fi
      #echo "${login_command[@]}" "https://almascience.eso.org/dataPortal/login"
      echo $("${download_command[@]}" "$file/client/status/$status")

      echo "        download $filename was interrupted with error code ${tool_name}/${status}"
      attempt_num=$[${attempt_num}+1]
      if [ ${attempt_num} -ge ${MAX_RETRIES} ]; then
        echo "        ERROR giving up on downloading $filename after ${MAX_RETRIES} attempts  - rerun the script manually to retry."
        download_status=1
      else
        echo "        download $filename will automatically resume after ${WAIT_SECS_BEFORE_RETRY} seconds"
        sleep ${WAIT_SECS_BEFORE_RETRY}
        echo "        resuming download of $filename, attempt $[${attempt_num}+1]"
      fi
    fi
  done
  exit ${download_status}
}
export -f dl

tars_exist=0
invalid_tars_exist=0
function check_integrity {
	for nextfile in ${LIST}; do
		tarname=$( basename ${nextfile} )
		if [[ ${nextfile} =~ ^.*\.tar$ ]] && [ -f ${tarname} ]; then
			tars_exist=1
			tar tf ${tarname} >/dev/null 2>&1
			tar_corrupted=$?
			if [[ ${tar_corrupted} -eq 1 ]]; then
				echo "tar invalid: ${tarname}"
				invalid_tars_exist=1
				break
			fi
		fi
	done
	if [[ ${invalid_tars_exist} -eq 1 ]]; then
		echo ""
		echo "Some of the downloads unfortunately failed. Please re-run the download script again in a few hours and, if this still fails, submit a ticket to the ALMA Helpdesk"
	fi
}

function unpack_tars {
	echo ""
	# override TIMEOUT so we never time-out. These files may take days to download. It's annoying if the user comes back
	# to the terminal after a weekend and finds that they weren't quick enough to automatically untar the files.
	# That's a pretty big timeout. One year in seconds.
	read -p "All files have been downloaded successfully. Should they be untarred? [Y/n] " -n 1 -r -t 31449600
	echo ""
	# ICT-19774 default should be 'Y', therefore the strange conditional logic "! [[ a =~ b ]]"
	if ! [[ ${REPLY} =~ ^[Nn]$ ]]; then
		read -p "Should they be untarred with the directory structure?                 [Y/n] " -n 1 -r -t 31449600 keep_dir_path
		echo ""
		for nextfile in ${LIST}; do
			if [[ ${nextfile} =~ ^.*\.tar$ ]]; then
				tarname=$( basename ${nextfile} )
				echo "	    unpacking ${tarname}"
				if ! [[ ${keep_dir_path} =~ ^[Nn]$ ]]; then 
					tar xf ${tarname}
				elif [[ ${tarname} =~ ^.*\.sdm.tar$ ]]; then
					# special case for ASDMs where we keep the ASDM dir structure, but nothing above the ASDM dir itself
					depth=$( tar -tf ${tarname} | grep "raw/$" | sed 's/\// /g' | wc -w | tr -d ' ' )
					tar --strip-components ${depth} -xf ${tarname}
				else
					# generic case for removing the directory path, and dumping all files in pwd
					tar_entries=$( tar -tf ${tarname} | grep -v '/$' )
					for next_entry in ${tar_entries}; do
						depth=$( echo "${next_entry}" | tr -cd \/ | wc -c | tr -d ' ' )
						tar --strip-components ${depth} -xf ${tarname} ${next_entry}
					done
				fi
			fi
		done
	fi
}

# Main body
# ---------

# check that we have one of the required download tools
if ! (command -v "wget" > /dev/null 2>&1 || command -v "curl" > /dev/null 2>&1); then
   echo "ERROR: neither 'wget' nor 'curl' are available on your computer. Please install one of them.";
   exit 1
fi

echo "Downloading the following files in up to 5 parallel streams. Total size is 69GB."
echo "${LIST}"
echo "In case of errors each download will be automatically resumed up to 3 times after a 5 minute delay"
echo "To manually resume interrupted downloads just re-run the script."
# tr converts spaces into newlines. Written legibly (spaces replaced by '_') we have: tr "\_"_"\\n"
# IMPORTANT. We still have infrastructure issues. Please don't split the individual downloads in parallel streams, and 
# please don't increase the P5 parameter.

# Included for retrials of passwords, for ticket : ICT-16332
login_attempts_remaining=3
while [ $login_attempts_remaining -gt 0 ]; do
    start_session
    if [ $authentication_status -eq 0 ]; then
        login_attempts_remaining=0
    else
        end_session
        login_attempts_remaining=$(( $login_attempts_remainig - 1 ))
    fi
done
all_downloads_status=0
if [ $authentication_status -eq 0 ]; then
	echo "your downloads will start shortly...."
	# "dl" is a function for downloading. I abbreviated the name to leave more space for the filename
	echo ${LIST[@]} | tr \  \\n | xargs -P5 -n1 -I '{}' bash -c 'dl {}' 2> /dev/null
	all_downloads_status=$?
fi
end_session
if [ $all_downloads_status -eq 0 ]; then
	check_integrity
	if [ $tars_exist -eq 1 ] && [ $invalid_tars_exist -eq 0 ]; then
		unpack_tars
	fi
	echo "Done."
else
	echo "Failed"
	exit 1
fi
# EOF
