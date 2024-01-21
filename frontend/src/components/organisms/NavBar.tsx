import DockerHub from "../../assets/svgs/docker.svg?react";
import Donation from "../../assets/svgs/donation.svg?react";
import GitHub from "../../assets/svgs/github.svg?react";
import Logo from "../../assets/svgs/logo.svg?react";
import { useNavigate } from "react-router-dom";

export default function NavBar() {
  const navigate = useNavigate();

  return (
    <nav
      className="sticky flex flex-col items-center justify-between border-r border-white border-opacity-50 px-2 py-3 
        max-md:flex-row max-md:border-b max-md:px-3 max-md:py-2 md:min-h-screen"
    >
      <Logo
        className="h-6 w-6 cursor-pointer fill-white"
        onClick={() => navigate("/app")}
      />
      <div className="flex flex-col items-center max-md:flex-row max-md:space-x-3 md:space-y-3">
        <a href={"/app"} target="_blank" rel="noopener noreferrer">
          <Donation className="h-5 w-5 fill-white" />
        </a>
        <a
          href={"https://hub.docker.com/u/ilovedatajjia"}
          target="_blank"
          rel="noopener noreferrer"
        >
          <DockerHub className="h-5 w-5 fill-white" />
        </a>
        <a
          href={"https://github.com/iLoveDataJjia"}
          target="_blank"
          rel="noopener noreferrer"
        >
          <GitHub className="h-5 w-5 fill-white" />
        </a>
      </div>
    </nav>
  );
}
